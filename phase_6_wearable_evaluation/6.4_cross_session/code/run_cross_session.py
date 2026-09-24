"""Offline cross-session subject-independent evaluation of 60-second SEED windows."""
import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

STAGE = Path(__file__).resolve().parents[1]
CALIBRATION = STAGE.parent / '6.3_calibration_length/code/run_calibration_length.py'
spec = importlib.util.spec_from_file_location('calibration63', CALIBRATION)
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)
base = cal.base
CONDITIONS = ['zero_cal', 'session1_cal']
PROTOCOL = 'primary'
METRICS = ['accuracy', 'macro_f1', 'weighted_f1', 'gain_pp']


def canonical_conditions(requested):
    base.check(bool(requested) and set(requested) <= set(CONDITIONS), 'Invalid conditions')
    # Preserve a contemporaneous matched-row zero anchor when calibration is requested.
    return [c for c in CONDITIONS if c in set(requested) | {'zero_cal'}]


def checked_splits(df):
    """All positions refer to df row order; validate every fold before fitting."""
    df = df.reset_index(drop=True)
    base.check(df.subject_num.nunique() >= 2, 'LOSO needs at least two subjects')
    base.check(not df.duplicated(base.KEY).any(), 'Duplicate window keys')
    base.check(df.session_index.isin([1, 2, 3]).all(), 'Unknown session index')
    plans = []
    for subject in sorted(df.subject_num.unique()):
        held = df.subject_num == subject
        train = np.flatnonzero(~held & df.session_index.isin([1, 2]))
        chosen = np.flatnonzero(held & (df.session_index == 1))
        test = np.flatnonzero(held & (df.session_index == 3))
        excluded = np.flatnonzero(held & (df.session_index == 2))
        base.check(len(chosen) > 0 and len(test) > 0, f'S{subject:02d} requires Session 1 and Session 3')
        base.check(len(train) > 0, 'Empty training fold')
        base.check(not set(df.iloc[train].subject_num) & set(df.iloc[test].subject_num), 'Subject overlap')
        base.check((df.iloc[chosen].subject_num == subject).all() and
                   (df.iloc[test].subject_num == subject).all(), 'Calibration/test subject mismatch')
        base.check((df.iloc[chosen].session_index == 1).all(), 'Calibration outside Session 1')
        base.check((df.iloc[test].session_index == 3).all(), 'Test outside Session 3')
        base.check(df.iloc[train].session_index.isin([1, 2]).all(), 'Training includes Session 3')
        base.check(not set(excluded) & (set(train) | set(chosen) | set(test)), 'Held-out Session 2 used')
        base.check(not set(chosen) & (set(train) | set(test)), 'Calibration overlap')
        base.check(set(df.iloc[train]['class']) == set(base.LABELS), 'Training fold requires all three classes')
        chosen = df.iloc[chosen].sort_values(cal.ORDER, kind='stable').index.to_numpy()
        plans.append((int(subject), train, chosen, test))
    return plans


def session_coverage(df):
    """Trial completeness supplements 6.2's validated full-window run coverage."""
    rows = []
    for subject, dates in base.cfg.SUBJECT_SESSIONS.items():
        for index, date in enumerate(dates, 1):
            block = df[(df.subject_num == subject) & (df.session_index == index)]
            found = set(block.trial)
            rows.append({'subject_num': subject, 'session_index': index, 'session': int(date),
                         'windows': len(block), 'passed_trials': len(found),
                         'missing_trials': sorted(set(range(1, 16)) - found),
                         'complete': found == set(range(1, 16))})
    return rows


def eligibility(df, audit, pilot):
    base.eligibility(df, audit, pilot)
    if not pilot:
        base.check(all(row['complete'] for row in session_coverage(df)),
                   'Final requires every expected trial in Sessions 1, 2 and 3')
    return checked_splits(df.reset_index(drop=True))


def split_manifest(df, plans, conditions, mode):
    parts = []
    for subject, train, chosen, test in plans:
        for condition in conditions:
            part = df[[c for c in base.META if c not in ['class', 'class_label']]].copy()
            roles = np.full(len(df), 'excluded_other_session3', dtype=object)
            held = df.subject_num.to_numpy() == subject
            roles[held & (df.session_index.to_numpy() == 2)] = 'excluded_heldout_session2'
            roles[train], roles[test] = 'train', 'test'
            roles[chosen] = 'calibration' if condition == 'session1_cal' else 'excluded_zero_cal_session1'
            part['role'] = roles
            part['held_out_subject'] = subject
            part['condition'], part['protocol'], part['mode'] = condition, PROTOCOL, mode
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def evaluate(df, params, mode, datasets=None, conditions=CONDITIONS, audit=None):
    df = df.reset_index(drop=True)
    sets = base.validate_features(df)
    datasets = list(sets) if datasets is None else list(datasets)
    base.check(bool(datasets) and len(set(datasets)) == len(datasets) and set(datasets) <= set(sets), 'Invalid datasets')
    base.check(mode in ['pilot', 'final'], 'Invalid mode')
    conditions = canonical_conditions(conditions)
    if mode == 'final':
        base.check(audit is not None, 'Final requires validated ingestion coverage')
        plans = eligibility(df, audit, False)
    else:
        plans = checked_splits(df)
    scores, predictions = [], []
    for name in datasets:
        X = df[sets[name]].to_numpy(dtype=float)
        for subject, train, chosen, test in plans:
            pipe = base.build_pipeline(len(sets[name]), params[name])
            pipe.fit(X[train], df.iloc[train]['class'].to_numpy())
            fold_scores = []
            for condition in conditions:
                # Only S1/S3 can reach calibration code; S2 is absent even from X_target.
                if condition == 'zero_cal':
                    target = X[test]
                    local_cal = np.array([], dtype=int)
                    local_test = np.arange(len(test))
                    cal_condition = '0'
                else:
                    target = X[np.concatenate([chosen, test])]
                    local_cal = np.arange(len(chosen))
                    local_test = np.arange(len(chosen), len(chosen) + len(test))
                    cal_condition = 'session1'
                pred, _, _ = cal.predict_target(pipe, target, local_cal, local_test, cal_condition)
                base.check(pred.shape == (len(test),) and np.isfinite(pred).all(), 'Invalid/incomplete predictions')
                truth = df.iloc[test]['class'].to_numpy()  # Scoring only, after inference.
                row = {'mode': mode, 'protocol': PROTOCOL, 'dataset': name, 'model': 'SVM_Tuned',
                       'subject_num': subject, 'subject_id': f'S{subject:02d}', 'condition': condition,
                       'train_session_set': '1,2', 'calibration_session': 0 if condition == 'zero_cal' else 1,
                       'test_session': 3, 'n_features': len(sets[name]), 'n_train_windows': len(train),
                       'n_calibration_windows': len(local_cal), 'n_test_windows': len(test),
                       'accuracy': accuracy_score(truth, pred),
                       'macro_f1': f1_score(truth, pred, labels=base.LABELS, average='macro', zero_division=0),
                       'weighted_f1': f1_score(truth, pred, labels=base.LABELS, average='weighted', zero_division=0),
                       'confusion_matrix': json.dumps(confusion_matrix(truth, pred, labels=base.LABELS).tolist())}
                fold_scores.append(row)
                part = df.iloc[test][base.META].copy()
                for key in ['mode', 'protocol', 'dataset', 'model', 'condition', 'train_session_set', 'calibration_session', 'test_session']:
                    part[key] = row[key]
                part['y_true'], part['y_pred'] = truth, pred
                predictions.append(part)
            zero = fold_scores[0]['accuracy']
            for row in fold_scores:
                row['gain_pp'] = 100 * (row['accuracy'] - zero)
            scores.extend(fold_scores)
    per = pd.DataFrame(scores)
    pred = pd.concat(predictions, ignore_index=True)
    expected = sum(len(test) for _, _, _, test in plans) * len(datasets) * len(conditions)
    base.check(len(pred) == expected and not pred.duplicated(['dataset', 'condition'] + base.KEY).any(),
               'Incomplete/duplicate predictions')
    summary = []
    for (name, condition), block in per.groupby(['dataset', 'condition'], sort=False):
        row = {'mode': mode, 'protocol': PROTOCOL, 'dataset': name, 'condition': condition,
               'train_session_set': '1,2', 'calibration_session': int(block.calibration_session.iloc[0]),
               'test_session': 3, 'n_subjects': len(block), 'n_test_windows': int(block.n_test_windows.sum())}
        for metric in METRICS:
            row[f'mean_{metric}'] = block[metric].mean()
            row[f'std_{metric}'] = block[metric].std(ddof=1)
        summary.append(row)
    return per, pd.DataFrame(summary), pred, split_manifest(df, plans, conditions, mode)


def save_results(results, output, figures):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for frame, suffix in zip(results, ['per_subject', 'summary', 'predictions', 'split_manifest']):
        frame.to_csv(output / f'cross_session_{suffix}.csv', index=False)
    summary = results[1]
    datasets = summary.dataset.unique().tolist()
    conditions = [c for c in CONDITIONS if c in set(summary.condition)]
    for metric, label in [('accuracy', 'Accuracy'), ('macro_f1', 'Macro F1')]:
        fig, ax = plt.subplots(figsize=(9, 5))
        width = .8 / len(conditions)
        for i, condition in enumerate(conditions):
            block = summary[summary.condition == condition].set_index('dataset').loc[datasets]
            ax.bar(np.arange(len(datasets)) + (i - (len(conditions)-1)/2)*width,
                   block[f'mean_{metric}'], width, yerr=block[f'std_{metric}'], capsize=4, label=condition)
        ax.set_xticks(np.arange(len(datasets)), datasets)
        ax.set(ylabel=f'{label} (subject mean +/- sample SD)', ylim=(0, 1),
               title=f'{summary["mode"].iloc[0].upper()}: primary, held-out Session-3 evaluation')
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f'cross_session_{metric}.png', dpi=150)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--pilot', action='store_true')
    modes.add_argument('--require-all-subjects', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--datasets', nargs='+', choices=list(base.PARAM_SOURCE), default=list(base.PARAM_SOURCE))
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=CONDITIONS)
    args = parser.parse_args(argv)
    mode = 'pilot' if args.pilot else 'final'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = STAGE / 'output' / ('audit' if args.audit_only else mode) / stamp
    output.mkdir(parents=True)
    report = {'status': 'running', 'mode': mode, 'protocol': PROTOCOL, 'audit_only': args.audit_only,
              'datasets': list(dict.fromkeys(args.datasets)), 'conditions': canonical_conditions(args.conditions),
              'train_sessions': [1, 2], 'calibration_session': 1, 'test_session': 3,
              'held_out_session2': 'excluded', 'ddof': 0, 'std_floor': cal.EPS,
              'training_transform': 'raw -> train StandardScaler -> selector -> SVM',
              'zero_cal_transform': 'raw Session3 -> train StandardScaler -> selector -> SVM',
              'session1_cal_transform': 'raw Session3 -> Session1 target z-score -> fitted selector -> fitted SVM',
              'parameter_mapping': base.PARAM_SOURCE,
              'limitations': 'Offline SEED simulation; historical hyperparameters tuned on the same cohort, not independent/nested selection.'}
    try:
        report.update(script_sha256=base.sha(__file__), calibration_script_sha256=base.sha(CALIBRATION),
                      baseline_script_sha256=base.sha(cal.BASELINE), config_sha256=base.sha(base.cfg.__file__),
                      parameter_sha256=base.sha(base.TUNING), parameters=base.fixed_parameters(),
                      feature_sets={k: v for k, v in base.feature_sets().items() if k in report['datasets']})
        df, audit = base.collect()
        audit['session_coverage'] = session_coverage(df)
        for pilot, label in [(True, 'pilot'), (False, 'final')]:
            try:
                eligibility(df, audit, pilot)
                audit[f'{label}_eligibility'] = {'eligible': True}
            except ValueError as error:
                audit[f'{label}_eligibility'] = {'eligible': False, 'reason': str(error)}
        cal.write_json(output / 'coverage_report.json', audit)
        print(f'Available: {audit["available_subjects"]}; complete: {audit["complete_subjects"]}; windows: {len(df)}')
        if args.audit_only:
            report['status'] = 'audit_only'
            cal.write_json(output / 'run_report.json', report)
            print(f'No classifier fitted. Audit: {output}')
            return 0
        eligibility(df, audit, args.pilot)
        cal.write_json(output / 'run_report.json', report)
        results = evaluate(df, report['parameters'], mode, report['datasets'], report['conditions'], audit)
        save_results(results, output, STAGE / 'figures' / mode / stamp)
        report['status'] = 'passed'
        cal.write_json(output / 'run_report.json', report)
        print(f'Results: {output}')
        return 0
    except (ValueError, KeyError, OSError) as error:
        report.update(status='rejected', reason=str(error))
        cal.write_json(output / 'run_report.json', report)
        print(f'Rejected: {error}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
