"""Zero-calibration windowed LOSO. Default: require all 15 complete subjects."""
import argparse
import ast
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[3]
STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as cfg

INPUT = STAGE.parent / '6.1_windowed_feature_extraction' / 'output'
TUNING = ROOT / 'phase_4_classification/4.6_hyperparameter_tuning/output/tuning_best_params.csv'
META = ['subject_id', 'subject_num', 'session', 'session_index', 'trial', 'class',
        'class_label', 'window_id', 'start_sample', 'stop_sample', 'start_seconds', 'stop_seconds']
KEY = ['subject_num', 'session', 'trial', 'window_id']
REGIONS = ['Prefrontal', 'Frontal', 'Temporal', 'Central', 'Parietal', 'Occipital']
BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'broadband']
PARAM_SOURCE = {'Spectral': 'Spectral', 'Connectivity': 'GC+PDC_ROI',
                'Fusion': 'GC+PDC_ROI+Spectral'}
LABELS = [0, 1, 2]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_sets():
    """Exact schema of 6.1; no dtype-based or label-based feature discovery."""
    spectral = sorted(f'spec_{metric}_{band}_{ch}' for metric in ['power', 'de']
                      for band in BANDS[:-1] for ch in cfg.CHANNEL_NAMES)
    suffixes = [f'intra_{a}' if a == b else f'{a}_to_{b}' for a in REGIONS for b in REGIONS]
    suffixes += [f'{direction}_strength_{r}' for direction in ['in', 'out'] for r in REGIONS]
    suffixes += ['mean_inter_strength', 'mean_intra_strength']
    connectivity = sorted(f'{prefix}_{suffix}' for prefix in
                          ['gc_roi'] + [f'pdc_{b}_roi' for b in BANDS] for suffix in suffixes)
    return {'Spectral': spectral, 'Connectivity': connectivity, 'Fusion': spectral + connectivity}


def validate_features(df):
    sets = feature_sets()
    expected = set(META + sets['Fusion'])
    check(set(df.columns) == expected and df.columns.is_unique,
          f'Unexpected schema; missing={sorted(expected - set(df.columns))}, '
          f'extra={sorted(set(df.columns) - expected)}')
    check(not set(META) & set(sets['Fusion']), 'Metadata in X')
    for name, columns in sets.items():
        check(len(columns) == {'Spectral': 620, 'Connectivity': 350, 'Fusion': 970}[name],
              f'Feature count mismatch: {name}')
    check(all(pd.api.types.is_numeric_dtype(df[c]) for c in sets['Fusion']), 'Non-numeric features')
    check(np.isfinite(df[sets['Fusion']].to_numpy(dtype=float)).all(), 'Non-finite features')
    return sets


def read_run(report):
    info = json.loads(report.read_text(encoding='utf-8'))
    check(info.get('status') == 'passed', 'Run not passed')
    folder = report.parent
    csv_path = folder / 'windowed_features.csv'
    with csv_path.open(encoding='utf-8', newline='') as stream:
        header = next(csv.reader(stream))
    check(len(header) == len(set(header)), f'Duplicate CSV columns: {csv_path}')
    df = pd.read_csv(csv_path)
    validate_features(df)
    check(len(df) > 0 and not df[META].isna().any().any(), 'Empty run or missing metadata')
    for col in ['subject_num', 'session', 'session_index', 'trial', 'class', 'window_id',
                'start_sample', 'stop_sample']:
        values = pd.to_numeric(df[col], errors='raise').to_numpy()
        check(np.isfinite(values).all() and np.equal(values, np.floor(values)).all(), f'Invalid {col}')
    check(len(df[['subject_num', 'session', 'trial']].drop_duplicates()) == 1, 'Mixed trials in one run')
    first = df.iloc[0]
    subject, session, trial = (int(first[c]) for c in ['subject_num', 'session', 'trial'])
    check(subject in cfg.SUBJECT_SESSIONS and 1 <= trial <= 15, 'Unknown subject/trial')
    dates = cfg.SUBJECT_SESSIONS[subject]
    check(str(session) in dates, 'Unknown session')
    check((df.subject_id == f'S{subject:02d}').all(), 'Subject ID mismatch')
    check((df.session_index == dates.index(str(session)) + 1).all(), 'Session index mismatch')
    raw_label = cfg.TRIAL_LABELS[trial - 1]
    check((df['class'] == cfg.LABEL_MAP[raw_label]).all() and
          (df.class_label == cfg.EMOTION_MAP[raw_label]).all(), 'Trial label mismatch')
    check(folder.parent.name == f'trial_{trial:02d}' and
          folder.parent.parent.name == f'session_{session}' and
          folder.parent.parent.parent.name == f'subject_{subject:02d}', 'Path/identity mismatch')
    check(info['fs'] == cfg.TARGET_FS and info['window_seconds'] == 60 and
          info['overlap_samples'] == 0 and info['tail_policy'] == 'drop', 'Window protocol mismatch')
    size = int(info['fs'] * 60)
    n = info['input_shape'][1] // size
    check(info['input_shape'][0] == 62 and info['channels'] == cfg.CHANNEL_NAMES, 'Channel mismatch')
    check(n == len(df) == info['n_windows'] == info['completed_windows'] and
          info['n_features'] == 970, 'Incomplete run')
    check(info['discarded_samples'] == info['input_shape'][1] % size, 'Tail mismatch')
    df = df.sort_values('window_id').reset_index(drop=True)
    check(df.window_id.tolist() == list(range(1, n + 1)), 'Duplicate/missing window IDs')
    starts = np.arange(n) * size
    check(np.array_equal(df.start_sample, starts) and np.array_equal(df.stop_sample, starts + size),
          'Window gap/overlap or invalid length')
    check(np.allclose(df.start_seconds, starts / info['fs']) and
          np.allclose(df.stop_seconds, (starts + size) / info['fs']), 'Time bounds mismatch')
    return (subject, session, trial), df, info


def collect(input_root=INPUT):
    """Select latest passed rerun per trial; never count duplicate windows twice."""
    candidates, ignored = {}, []
    for report in sorted(Path(input_root).glob('subject_*/session_*/trial_*/run_*/sanity_report.json')):
        info = json.loads(report.read_text(encoding='utf-8'))
        if info.get('status') != 'passed':
            ignored.append(str(report))
            continue
        candidates.setdefault(report.parent.parent, []).append(report)
    frames, selected, superseded, signature = [], [], [], None
    for reports in candidates.values():
        report = sorted(reports)[-1]
        superseded.extend(str(p) for p in sorted(reports)[:-1])
        key, df, info = read_run(report)
        protocol = {k: info[k] for k in ['pipeline_sha256', 'fs', 'channels', 'gc_max_lag',
                    'gc_threshold', 'gc_alpha', 'pdc_max_order', 'pdc_min_order', 'bands',
                    'matrix_orientation', 'normalization']}
        fingerprint = json.dumps(protocol, sort_keys=True)
        check(signature is None or signature == fingerprint, 'Mixed 6.1 pipeline versions/protocols')
        signature = fingerprint
        frames.append(df)
        selected.append({'subject_num': key[0], 'session': key[1], 'trial': key[2],
                         'report': str(report), 'report_sha256': sha(report),
                         'csv_sha256': sha(report.parent / 'windowed_features.csv'), 'windows': len(df)})
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=META + feature_sets()['Fusion'])
    check(not data.duplicated(KEY).any(), 'Duplicate window keys')
    coverage = []
    for subject, dates in cfg.SUBJECT_SESSIONS.items():
        expected = {(subject, int(date), trial) for date in dates for trial in range(1, 16)}
        found = {(r['subject_num'], r['session'], r['trial']) for r in selected if r['subject_num'] == subject}
        coverage.append({'subject_num': subject, 'subject_id': f'S{subject:02d}',
                         'passed_trials': len(found), 'expected_trials': len(expected),
                         'complete': found == expected,
                         'windows': int((data.subject_num == subject).sum()),
                         'missing_trials': [f'{date}/trial_{trial:02d}' for _, date, trial in sorted(expected - found)]})
    audit = {'coverage': coverage, 'selected_runs': selected, 'ignored_nonpassed': ignored,
             'superseded_passed': superseded, 'available_subjects': sorted(data.subject_num.unique().tolist()),
             'complete_subjects': [r['subject_num'] for r in coverage if r['complete']]}
    return data, audit


def eligibility(df, audit, pilot=False):
    if not pilot:
        check(set(audit['complete_subjects']) == set(cfg.SUBJECT_SESSIONS),
              'Final evaluation requires 15 COMPLETE subjects (45 passed trials each). Use --pilot for partial coverage.')
    check(df.subject_num.nunique() >= 2, 'LOSO needs at least two available subjects, including in --pilot mode.')


def fixed_parameters(path=TUNING):
    table = pd.read_csv(path)
    result = {}
    for name, source in PARAM_SOURCE.items():
        rows = table[(table.Dataset == source) & (table.Model == 'SVM_Tuned')]
        check(len(rows) == 1, f'Expected one SVM_Tuned parameter row: {source}')
        params = ast.literal_eval(rows.iloc[0].Best_Params)
        check(set(params) == {'clf__C', 'clf__gamma'}, 'Unsupported historical parameter schema')
        check(float(params['clf__C']) > 0, 'Invalid C')
        gamma = params['clf__gamma']
        check(gamma in ['scale', 'auto'] if isinstance(gamma, str) else float(gamma) > 0, 'Invalid gamma')
        result[name] = {'C': params['clf__C'], 'gamma': gamma}
    return result


def build_pipeline(n_features, params):
    return Pipeline([('scaler', StandardScaler()),
                     ('select', SelectKBest(f_classif, k=min(1200, n_features))),
                     ('clf', SVC(kernel='rbf', probability=False, **params))])


def checked_splits(df):
    groups, y = df.subject_num.to_numpy(), df['class'].to_numpy()
    check(len(np.unique(groups)) >= 2, 'LOSO needs at least two subjects')
    splits = list(LeaveOneGroupOut().split(df, y, groups))
    for train, test in splits:
        check(not set(groups[train]) & set(groups[test]), 'Train/test subject overlap')
        check(len(np.unique(groups[test])) == 1, 'Test fold contains multiple subjects')
        check(set(y[train]) == set(LABELS), 'Every training fold must contain all three classes')
    return splits


def evaluate(df, params, mode, observer=None):
    sets = validate_features(df)
    splits = checked_splits(df)  # Validate every fold before fitting any model.
    scores, predictions = [], []
    y = df['class'].to_numpy()
    for name, columns in sets.items():
        X = df[columns].to_numpy(dtype=float)
        for train, test in splits:
            pipe = build_pipeline(len(columns), params[name])
            pipe.fit(X[train], y[train])
            check(pipe.n_features_in_ == len(columns), 'Fold feature count mismatch')
            if observer is not None:
                observer(name, pipe, train, test)
            pred = pipe.predict(X[test])
            subject = int(df.iloc[test[0]].subject_num)
            scores.append({'mode': mode, 'dataset': name, 'model': 'SVM_Tuned',
                           'subject_num': subject, 'subject_id': f'S{subject:02d}',
                           'n_train_windows': len(train), 'n_test_windows': len(test),
                           'n_features': len(columns), 'selected_features': int(pipe['select'].get_support().sum()),
                           'accuracy': accuracy_score(y[test], pred),
                           'macro_f1': f1_score(y[test], pred, labels=LABELS, average='macro', zero_division=0),
                           'weighted_f1': f1_score(y[test], pred, labels=LABELS, average='weighted', zero_division=0),
                           'confusion_matrix': json.dumps(confusion_matrix(y[test], pred, labels=LABELS).tolist())})
            part = df.iloc[test][META].copy()
            part['dataset'], part['mode'], part['model'] = name, mode, 'SVM_Tuned'
            part['y_true'], part['y_pred'] = y[test], pred
            predictions.append(part)
    per_subject = pd.DataFrame(scores)
    predictions = pd.concat(predictions, ignore_index=True)
    summary = per_subject.groupby(['mode', 'dataset', 'model'], sort=False).agg(
        n_subjects=('subject_num', 'nunique'), n_test_windows=('n_test_windows', 'sum'),
        n_features=('n_features', 'first'), mean_accuracy=('accuracy', 'mean'),
        std_accuracy=('accuracy', 'std'), mean_macro_f1=('macro_f1', 'mean'),
        std_macro_f1=('macro_f1', 'std'), mean_weighted_f1=('weighted_f1', 'mean'),
        std_weighted_f1=('weighted_f1', 'std')).reset_index()
    check(len(predictions) == 3 * len(df) and not predictions.duplicated(['dataset'] + KEY).any(),
          'Incomplete/duplicate out-of-fold predictions')
    return per_subject, summary, predictions


def save_results(results, output, figures):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    scores, summary, predictions = results
    for frame, name in zip(results, ['windowed_loso_per_subject.csv', 'windowed_loso_summary.csv', 'windowed_predictions.csv']):
        frame.to_csv(output / name, index=False)
    mode = scores['mode'].iloc[0].upper()
    fig, ax = plt.subplots(figsize=(12, 5))
    scores.pivot(index='subject_id', columns='dataset', values='accuracy')[list(PARAM_SOURCE)].plot.bar(ax=ax)
    ax.set(ylabel='Accuracy', ylim=(0, 1), title=f'{mode}: zero-calibration LOSO per subject')
    fig.tight_layout()
    fig.savefig(figures / 'per_subject_accuracy.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    summary = summary.set_index('dataset').loc[list(PARAM_SOURCE)]
    ax.bar(summary.index, summary.mean_accuracy, yerr=summary.std_accuracy, capsize=4)
    ax.set(ylabel='Mean subject accuracy (SD across subjects)', ylim=(0, 1),
           title=f'{mode}: Spectral vs Connectivity vs Fusion')
    fig.tight_layout()
    fig.savefig(figures / 'dataset_comparison.png', dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--pilot', action='store_true', help='Allow partial subjects/trials; non-final results')
    modes.add_argument('--require-all-subjects', action='store_true', help='Require all 15 complete subjects (default)')
    parser.add_argument('--audit-only', action='store_true', help='Validate inputs/coverage only; never fit a classifier')
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    mode = 'pilot' if args.pilot else 'final'
    output = STAGE / 'output' / ('audit' if args.audit_only else mode) / stamp
    output.mkdir(parents=True)
    try:
        df, audit = collect()
        (output / 'coverage_report.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        print(f'Available subjects: {audit["available_subjects"]}; complete: {audit["complete_subjects"]}; windows: {len(df)}')
        params = fixed_parameters()
        if args.audit_only:
            print(f'Audit only: no model fitted. Report: {output}')
            return 0
        eligibility(df, audit, args.pilot)
        provenance = {'status': 'running', 'mode': mode, 'zero_calibration': True,
                      'parameter_mapping': PARAM_SOURCE, 'parameters': params, 'parameter_file': str(TUNING),
                      'parameter_sha256': sha(TUNING), 'script_sha256': sha(__file__),
                      'feature_sets': feature_sets(), 'normalization': 'StandardScaler fitted on training fold only',
                      'historical_tuning_limitation': 'Parameters previously tuned on the same SEED cohort; not independent/nested model selection.'}
        run_report = output / 'run_report.json'
        run_report.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
        df.to_csv(output / 'merged_windowed_features.csv', index=False)
        results = evaluate(df, params, mode)
        save_results(results, output, STAGE / 'figures' / mode / stamp)
        provenance['status'] = 'passed'
        run_report.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
        print(f'Results: {output}')
        return 0
    except (ValueError, KeyError, OSError) as error:
        (output / 'failure.json').write_text(json.dumps({'status': 'rejected', 'reason': str(error)}, indent=2), encoding='utf-8')
        print(f'Rejected: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
