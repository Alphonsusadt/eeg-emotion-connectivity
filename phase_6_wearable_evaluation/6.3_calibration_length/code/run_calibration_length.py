"""Unlabeled target-only calibration; Phase 6.2 remains the training protocol."""
import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

STAGE = Path(__file__).resolve().parents[1]
BASELINE = STAGE.parent / '6.2_windowed_baseline/code/run_windowed_loso.py'
spec = importlib.util.spec_from_file_location('baseline62', BASELINE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
CONDITIONS = ['0', '1', '2', '5', '10', 'session1', 'full']
ORDER = ['subject_num', 'session_index', 'trial', 'window_id']
EPS = 1e-10


def canonical_conditions(requested):
    base.check(bool(requested) and set(requested) <= set(CONDITIONS), 'Unknown/empty conditions')
    # Anchors are always required for gains and the 6.2 comparison.
    return [c for c in CONDITIONS if c in set(requested) | {'0', 'full'}]


def select_rows(metadata, condition):
    """Only chronological metadata accepted. Return local positional indices."""
    base.check(set(metadata.columns) == set(ORDER), 'Selection accepts chronology metadata only, no labels/features')
    base.check(metadata.subject_num.nunique() == 1, 'Calibration must belong to one held-out subject')
    base.check(not metadata.duplicated(ORDER).any(), 'Duplicate chronology keys')
    base.check(condition in CONDITIONS, 'Unknown condition')
    meta = metadata.reset_index(drop=True)
    ordered = meta.sort_values(ORDER[1:], kind='stable').index.to_numpy()
    if condition == '0':
        cal = np.array([], dtype=int)
    elif condition == 'full':
        cal = ordered
    elif condition == 'session1':
        cal = ordered[meta.iloc[ordered].session_index.to_numpy() == 1]
        base.check(len(cal) > 0, 'Session 1 has no calibration windows')
    else:
        count = int(condition)
        base.check(len(ordered) > count, f'{condition} min requires >{count} held-out windows')
        cal = ordered[:count]
    evaluation = np.arange(len(meta)) if condition in ['0', 'full'] else np.setdiff1d(np.arange(len(meta)), cal)
    base.check(len(evaluation) > 0, f'Empty evaluation set for {condition}')
    if condition not in ['0', 'full']:
        base.check(not set(cal) & set(evaluation), 'Calibration/scoring overlap')
    return cal, evaluation


def transform_target(X, calibration, evaluation):
    """Population std (ddof=0); empty calibration computes no target statistics."""
    base.check(np.isfinite(X).all(), 'Non-finite raw values')
    if not len(calibration):
        return X[evaluation].copy(), None, None
    mu = X[calibration].mean(axis=0)
    sigma = X[calibration].std(axis=0, ddof=0)
    sigma = np.where(sigma < EPS, 1.0, sigma)
    transformed = (X[evaluation] - mu) / sigma
    base.check(np.isfinite(mu).all() and np.isfinite(sigma).all() and np.isfinite(transformed).all(),
               'Non-finite calibrated values')
    return transformed, mu, sigma


def predict_target(pipe, X, calibration, evaluation, condition):
    """X uses the same ordered feature columns as training; never refit any step."""
    base.check(X.ndim == 2 and X.shape[1] == pipe.n_features_in_, 'Target feature shape mismatch')
    base.check(condition in CONDITIONS, 'Unknown condition')
    base.check((condition == '0') == (len(calibration) == 0), 'Condition/calibration mismatch')
    transformed, mu, sigma = transform_target(X, calibration, evaluation)
    if condition == '0':
        # Preserve the Phase 6.2 raw-target prediction path exactly.
        pred = pipe.predict(transformed)
    else:
        # Target z-scores already occupy the selector's input coordinate space.
        # Applying the raw-training scaler here would normalize them twice.
        selected = pipe['select'].transform(transformed)
        base.check(np.isfinite(selected).all(), 'Non-finite selected target values')
        pred = pipe['clf'].predict(selected)
    return pred, mu, sigma


def gain_metrics(acc, zero, full):
    denominator = full - zero
    return {'gain_pp': 100 * (acc - zero), 'gap_to_full_pp': 100 * (full - acc),
            'recovery_pct': 100 * (acc - zero) / denominator if denominator > EPS else np.nan}


def condition_audit(df, conditions):
    rows = []
    for subject, frame in df.groupby('subject_num', sort=True):
        for condition in conditions:
            row = {'subject_num': int(subject), 'condition': condition}
            try:
                cal, ev = select_rows(frame[ORDER], condition)
                row.update(valid=True, calibration_windows=len(cal), evaluation_windows=len(ev))
            except ValueError as error:
                row.update(valid=False, reason=str(error))
            rows.append(row)
    return rows


def evaluate(df, params, mode, datasets=None, conditions=CONDITIONS, observer=None):
    df = df.reset_index(drop=True)
    sets = base.validate_features(df)
    datasets = list(sets) if datasets is None else datasets
    base.check(bool(datasets) and len(set(datasets)) == len(datasets) and set(datasets) <= set(sets), 'Invalid datasets')
    conditions = canonical_conditions(conditions)
    splits = base.checked_splits(df)
    plans = {}
    # Fail before fitting if any requested fold/condition is unavailable; no silent skipping.
    for fold, (train, test) in enumerate(splits):
        for condition in conditions:
            cal, ev = select_rows(df.iloc[test][ORDER], condition)
            base.check(not set(test[cal]) & set(train), 'Calibration rows in classifier training')
            plans[fold, condition] = cal, ev
    scores, predictions, selections = [], [], []
    for name in datasets:
        columns = sets[name]
        X = df[columns].to_numpy(dtype=float)
        for fold, (train, test) in enumerate(splits):
            pipe = base.build_pipeline(len(columns), params[name])
            # No subject-level transform of training features, exactly as in 6.2.
            pipe.fit(X[train], df.iloc[train]['class'].to_numpy())
            base.check(pipe.n_features_in_ == len(columns), 'Feature count mismatch')
            held = df.iloc[test]
            subject = int(held.subject_num.iloc[0])
            cached = {}
            for condition in conditions:
                cal, ev = plans[fold, condition]
                pred, mu, sigma = predict_target(pipe, X[test], cal, ev, condition)
                if observer is not None:
                    observer(name, subject, condition, pipe, train, test, cal, ev, mu, sigma)
                if condition == '0':
                    base.check(np.array_equal(pred, pipe.predict(X[test])), '0-min prediction differs from 6.2 pipeline')
                cached[condition] = (pred, ev)
                # Held-out labels are accessed only after prediction, for evaluation scoring.
                truth = held.iloc[ev]['class'].to_numpy()
                score = {'mode': mode, 'dataset': name, 'model': 'SVM_Tuned', 'subject_num': subject,
                         'subject_id': f'S{subject:02d}', 'condition': condition,
                         'oracle_reference': condition == 'full', 'n_features': len(columns),
                         'n_train_windows': len(train), 'calibration_window_count': len(cal),
                         'calibration_duration_minutes': float(len(cal)), 'evaluation_window_count': len(ev),
                         'accuracy': accuracy_score(truth, pred),
                         'macro_f1': f1_score(truth, pred, labels=base.LABELS, average='macro', zero_division=0),
                         'weighted_f1': f1_score(truth, pred, labels=base.LABELS, average='weighted', zero_division=0),
                         'confusion_matrix': json.dumps(confusion_matrix(truth, pred, labels=base.LABELS).tolist())}
                scores.append(score)
                part = held.iloc[ev][base.META].copy()
                part['dataset'], part['condition'], part['mode'] = name, condition, mode
                part['oracle_reference'] = condition == 'full'
                part['y_true'], part['y_pred'] = truth, pred
                predictions.append(part)
                if name == datasets[0]:
                    chosen = held.iloc[cal][ORDER].copy()
                    chosen['condition'] = condition
                    selections.append(chosen)
            zero = next(r['accuracy'] for r in scores if r['dataset'] == name and r['subject_num'] == subject and r['condition'] == '0')
            full = next(r['accuracy'] for r in scores if r['dataset'] == name and r['subject_num'] == subject and r['condition'] == 'full')
            for score in scores:
                if score['dataset'] != name or score['subject_num'] != subject:
                    continue
                score.update(gain_metrics(score['accuracy'], zero, full))
                _, ev = cached[score['condition']]
                truth = held.iloc[ev]['class'].to_numpy()
                matched_zero = accuracy_score(truth, cached['0'][0][ev])
                matched_full = accuracy_score(truth, cached['full'][0][ev])
                score.update(accuracy_0_same_rows=matched_zero, accuracy_full_same_rows=matched_full)
                score.update({f'matched_{k}': v for k, v in gain_metrics(score['accuracy'], matched_zero, matched_full).items()})
    per = pd.DataFrame(scores)
    pred = pd.concat(predictions, ignore_index=True)
    base.check(not pred.duplicated(['dataset', 'condition'] + base.KEY).any(), 'Duplicate evaluation predictions')
    summary_rows = []
    metrics = ['accuracy', 'macro_f1', 'weighted_f1', 'gain_pp', 'gap_to_full_pp', 'recovery_pct',
               'matched_gain_pp', 'matched_gap_to_full_pp', 'matched_recovery_pct']
    for name in datasets:
        for condition in conditions:
            block = per[(per.dataset == name) & (per.condition == condition)]
            row = {'mode': mode, 'dataset': name, 'condition': condition, 'condition_order': CONDITIONS.index(condition),
                   'oracle_reference': condition == 'full', 'n_subjects': len(block),
                   'mean_calibration_duration_minutes': block.calibration_duration_minutes.mean(),
                   'evaluation_windows_total': int(block.evaluation_window_count.sum())}
            for metric in metrics:
                row[f'mean_{metric}'] = block[metric].mean()
                row[f'std_{metric}'] = block[metric].std(ddof=1)
                row[f'n_valid_{metric}'] = int(block[metric].notna().sum())
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    curve = summary.copy()  # Explicit ordered, subject-weighted table for plotting.
    return per, summary, pred, curve, pd.concat(selections, ignore_index=True)


def save_results(results, output, figures):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for frame, name in zip(results, ['calibration_per_subject.csv', 'calibration_summary.csv',
                                   'calibration_predictions.csv', 'calibration_curve.csv', 'calibration_selection.csv']):
        frame.to_csv(output / name, index=False)
    summary = results[1]
    available = [c for c in CONDITIONS if c in set(summary.condition)]
    for metric, ylabel in [('accuracy', 'Mean subject accuracy'), ('macro_f1', 'Mean subject macro F1')]:
        for suffix, order in [('', [c for c in available if c != 'session1']), ('_all_conditions', available)]:
            fig, ax = plt.subplots(figsize=(10, 5))
            for name in summary.dataset.unique():
                rows = summary[summary.dataset == name].set_index('condition').loc[order]
                ax.errorbar(np.arange(len(order)), rows[f'mean_{metric}'],
                            yerr=rows[f'std_{metric}'], marker='o', capsize=3, label=name)
            ax.set_xticks(np.arange(len(order)), ['full\n(oracle)' if c == 'full' else c for c in order])
            ax.set(xlabel='Unlabeled calibration (minutes; session1/full are references)', ylabel=ylabel,
                   ylim=(0, 1), title=f'{summary["mode"].iloc[0].upper()}: target-only calibration (mean ± subject SD)')
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures / f'{metric}_vs_calibration{suffix}.png', dpi=150)
            plt.close(fig)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--pilot', action='store_true')
    modes.add_argument('--require-all-subjects', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--datasets', nargs='+', choices=list(base.PARAM_SOURCE), default=list(base.PARAM_SOURCE))
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=CONDITIONS)
    args = parser.parse_args(argv)
    conditions = canonical_conditions(args.conditions)
    datasets = list(dict.fromkeys(args.datasets))
    mode = 'pilot' if args.pilot else 'final'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = STAGE / 'output' / ('audit' if args.audit_only else mode) / stamp
    out.mkdir(parents=True)
    report = {'status': 'running', 'mode': mode, 'audit_only': args.audit_only,
              'datasets': datasets, 'conditions': conditions, 'ddof': 0, 'std_floor': EPS,
              'training_subject_normalization': False, 'calibration_space': 'raw features -> target z-score -> fitted selector -> fitted SVM; bypass train scaler',
              'oracle_condition': 'full', 'reference_name': 'full-subject oracle/reference normalization',
              'base_script_sha256': base.sha(BASELINE), 'script_sha256': base.sha(__file__),
              'parameter_sha256': base.sha(base.TUNING), 'parameter_mapping': base.PARAM_SOURCE,
              'historical_tuning_limitation': 'Fixed parameters previously tuned on the same SEED cohort, not independent/nested tuning.',
              'comparison_limitation': 'Limited conditions score fewer rows; use matched_* metrics for identical evaluation rows.'}
    try:
        df, audit = base.collect()
        audit['condition_availability'] = condition_audit(df, conditions)
        write_json(out / 'coverage_report.json', audit)
        report['parameters'] = base.fixed_parameters()
        report['feature_sets'] = {k: v for k, v in base.feature_sets().items() if k in datasets}
        print(f'Available: {audit["available_subjects"]}; complete: {audit["complete_subjects"]}; windows: {len(df)}')
        if args.audit_only:
            report['status'] = 'audit_only'
            write_json(out / 'run_report.json', report)
            print(f'No classifier fitted. Audit: {out}')
            return 0
        base.eligibility(df, audit, args.pilot)
        write_json(out / 'run_report.json', report)
        results = evaluate(df, report['parameters'], mode, datasets, conditions)
        save_results(results, out, STAGE / 'figures' / mode / stamp)
        report['status'] = 'passed'
        write_json(out / 'run_report.json', report)
        print(f'Results: {out}')
        return 0
    except (ValueError, KeyError, OSError) as error:
        report.update(status='rejected', reason=str(error))
        write_json(out / 'run_report.json', report)
        print(f'Rejected: {error}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
