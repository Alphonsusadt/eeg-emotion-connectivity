"""Causal streaming normalization: offline prequential evaluation, fixed classifier."""
import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

STAGE = Path(__file__).resolve().parents[1]
CROSS_SESSION = STAGE.parent / '6.4_cross_session/code/run_cross_session.py'
spec = importlib.util.spec_from_file_location('cross_session64', CROSS_SESSION)
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)
base, cal = cs.base, cs.cal
CONDITIONS = ['zero_cal', 'fixed_session1', 'streaming_running', 'streaming_ema']
DEFAULT_ALPHA = 0.1
METRICS = ['accuracy', 'macro_f1', 'weighted_f1', 'gain_vs_zero_pp', 'gain_vs_fixed_pp']


def validate_alpha(alpha):
    base.check(np.isfinite(alpha) and 0 < alpha <= 1, 'EMA alpha must be finite and in (0, 1]')


def canonical_conditions(requested):
    base.check(bool(requested) and set(requested) <= set(CONDITIONS), 'Invalid conditions')
    wanted = set(requested) | {'zero_cal'}
    if wanted & {'streaming_running', 'streaming_ema'}:
        wanted.add('fixed_session1')
    return [c for c in CONDITIONS if c in wanted]


def initial_moments(X):
    X = np.asarray(X, dtype=float)
    base.check(X.ndim == 2 and len(X) > 0 and X.shape[1] > 0, 'Session-1 initialization required')
    base.check(np.isfinite(X).all(), 'Non-finite initialization')
    mean = X.mean(axis=0)
    M2 = np.sum((X - mean)**2, axis=0)
    base.check(np.isfinite(mean).all() and np.isfinite(M2).all(), 'Non-finite initial moments')
    return len(X), mean, M2


def safe_std(variance):
    base.check(np.isfinite(variance).all(), 'Non-finite variance')
    # Clamp only possible negative roundoff; never store the std fallback as variance.
    std = np.sqrt(np.maximum(variance, 0.0))
    return np.where(std < cal.EPS, 1.0, std)


class RunningStatistics:
    """Welford population moments; O(n_features) state, no stored sample history."""
    def __init__(self, initialization):
        self.count, self.mean, self.M2 = initial_moments(initialization)

    def std(self):
        return safe_std(self.M2 / self.count)

    def update(self, x):
        x = np.asarray(x, dtype=float)
        base.check(x.shape == self.mean.shape and np.isfinite(x).all(), 'Invalid update row')
        delta = x - self.mean
        self.count += 1
        self.mean = self.mean + delta / self.count
        self.M2 = self.M2 + delta * (x - self.mean)
        base.check(np.isfinite(self.mean).all() and np.isfinite(self.M2).all(), 'Non-finite running moments')


class EMAStatistics:
    """S1 batch initialization followed by fixed-alpha central-moment updates."""
    def __init__(self, initialization, alpha=DEFAULT_ALPHA):
        validate_alpha(alpha)
        self.count, self.mean, M2 = initial_moments(initialization)
        self.variance = M2 / self.count
        self.alpha = alpha

    def std(self):
        return safe_std(self.variance)

    def update(self, x):
        x = np.asarray(x, dtype=float)
        base.check(x.shape == self.mean.shape and np.isfinite(x).all(), 'Invalid update row')
        delta = x - self.mean  # Must use the OLD mean for both recurrences.
        self.variance = (1 - self.alpha) * (self.variance + self.alpha * delta**2)
        self.mean = self.mean + self.alpha * delta
        self.count += 1  # Observed count, not effective sample size.
        base.check(np.isfinite(self.mean).all() and np.isfinite(self.variance).all(), 'Non-finite EMA moments')


def predict_stream(pipe, initialization, stream, condition, alpha=DEFAULT_ALPHA, observer=None):
    """Feature arrays only, already ordered. Observer is a test/debug hook after predict, before update."""
    base.check(condition in CONDITIONS, 'Invalid condition')
    validate_alpha(alpha)
    stream = np.asarray(stream, dtype=float)
    base.check(stream.ndim == 2 and len(stream) > 0 and stream.shape[1] == pipe.n_features_in_,
               'Stream feature shape mismatch')
    base.check(np.isfinite(stream).all(), 'Non-finite stream')
    if condition == 'zero_cal':
        # No initialization statistics or access to initialization rows.
        pred, _, _ = cal.predict_target(pipe, stream, np.array([], dtype=int), np.arange(len(stream)), '0')
        return pred, np.zeros(len(stream), dtype=int)
    initialization = np.asarray(initialization, dtype=float)
    base.check(initialization.ndim == 2 and len(initialization) > 0 and
               initialization.shape[1] == pipe.n_features_in_, 'Session-1 initialization required; no fallback')
    if condition == 'fixed_session1':
        target = np.concatenate([initialization, stream])
        n = len(initialization)
        pred, _, _ = cal.predict_target(pipe, target, np.arange(n), np.arange(n, len(target)), 'session1')
        return pred, np.full(len(stream), n, dtype=int)
    stats = RunningStatistics(initialization) if condition == 'streaming_running' else EMAStatistics(initialization, alpha)
    predictions, counts = [], []
    for index, x in enumerate(stream):
        base.check(stats.count == len(initialization) + index, 'Current/future window in history')
        count, mean, std = stats.count, stats.mean.copy(), stats.std()
        z = ((x - mean) / std).reshape(1, -1)
        base.check(np.isfinite(z).all(), 'Non-finite normalized target')
        selected = pipe['select'].transform(z)
        base.check(np.isfinite(selected).all(), 'Non-finite selected target')
        pred = pipe['clf'].predict(selected)
        base.check(pred.shape == (1,) and np.isfinite(pred).all(), 'Invalid stream prediction')
        predictions.append(pred[0])
        counts.append(count)
        if observer is not None:
            observer(index, count, mean, std.copy(), z.copy(), pred.copy())
        # Strict prequential order: only now can x enter target statistics.
        stats.update(x)
    return np.asarray(predictions), np.asarray(counts, dtype=int)


def chronological_plans(df, plans):
    """Keep inherited train/init positions; order only the held-out S3 stream."""
    return [(subject, train, chosen, df.iloc[test].sort_values(['trial', 'window_id'], kind='stable').index.to_numpy())
            for subject, train, chosen, test in plans]


def split_manifest(df, plans, conditions, mode):
    # Reuse 6.4's complete row-role assignment, including excluded held-out S2.
    inherited = cs.split_manifest(df, plans, ['zero_cal', 'session1_cal'], mode)
    parts = []
    for condition in conditions:
        source = 'zero_cal' if condition == 'zero_cal' else 'session1_cal'
        part = inherited[inherited.condition == source].copy()
        part['condition'] = condition
        part['role'] = part.role.replace({'calibration': 'initialization'})
        part['chronological_index'] = 0  # Non-test rows have no stream index.
        for subject, _, _, test in plans:
            mask = (part.held_out_subject == subject) & (part.role == 'test')
            order = {tuple(key): i for i, key in enumerate(df.iloc[test][base.KEY].to_numpy(), 1)}
            part.loc[mask, 'chronological_index'] = [order[tuple(key)] for key in part.loc[mask, base.KEY].to_numpy()]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def cumulative_curve(predictions):
    # Equal weight per subject still observed at index t; counts make tail attrition explicit.
    return predictions.groupby(['mode', 'protocol', 'dataset', 'condition', 'chronological_index'], sort=False).agg(
        mean_cumulative_accuracy=('cumulative_accuracy', 'mean'),
        std_cumulative_accuracy=('cumulative_accuracy', 'std'),
        n_subjects=('subject_num', 'nunique')).reset_index()


def evaluate(df, params, mode, datasets=None, conditions=CONDITIONS, alpha=DEFAULT_ALPHA, audit=None):
    validate_alpha(alpha)
    df = df.reset_index(drop=True)
    sets = base.validate_features(df)
    datasets = list(sets) if datasets is None else list(datasets)
    base.check(bool(datasets) and len(set(datasets)) == len(datasets) and set(datasets) <= set(sets), 'Invalid datasets')
    base.check(mode in ['pilot', 'final'], 'Invalid mode')
    conditions = canonical_conditions(conditions)
    if mode == 'final':
        base.check(audit is not None, 'Final requires validated ingestion coverage')
        plans = cs.eligibility(df, audit, False)
    else:
        plans = cs.checked_splits(df)
    plans = chronological_plans(df, plans)
    scores, predictions = [], []
    for name in datasets:
        X = df[sets[name]].to_numpy(dtype=float)
        for subject, train, chosen, test in plans:
            pipe = base.build_pipeline(len(sets[name]), params[name])
            pipe.fit(X[train], df.iloc[train]['class'].to_numpy())
            fold = []
            for condition in conditions:
                initialization = None if condition == 'zero_cal' else X[chosen]
                pred, history = predict_stream(pipe, initialization, X[test], condition, alpha)
                base.check(pred.shape == (len(test),) and np.isfinite(pred).all(), 'Invalid/incomplete predictions')
                n_init = 0 if condition == 'zero_cal' else len(chosen)
                expected_history = n_init + np.arange(len(test)) if condition.startswith('streaming_') else np.full(len(test), n_init)
                base.check(np.array_equal(history, expected_history), 'Incorrect pre-prediction history count')
                truth = df.iloc[test]['class'].to_numpy()  # Scoring after the complete label-free inference pass.
                row = {'mode': mode, 'protocol': cs.PROTOCOL, 'dataset': name, 'model': 'SVM_Tuned',
                       'subject_num': subject, 'subject_id': f'S{subject:02d}', 'condition': condition,
                       'train_session_set': '1,2', 'initialization_session': 0 if condition == 'zero_cal' else 1,
                       'test_session': 3, 'ema_alpha': alpha, 'ema_alpha_fixed': True,
                       'n_features': len(sets[name]), 'n_train_windows': len(train),
                       'n_initialization_windows': n_init, 'n_test_windows': len(test),
                       'accuracy': accuracy_score(truth, pred),
                       'macro_f1': f1_score(truth, pred, labels=base.LABELS, average='macro', zero_division=0),
                       'weighted_f1': f1_score(truth, pred, labels=base.LABELS, average='weighted', zero_division=0),
                       'confusion_matrix': json.dumps(confusion_matrix(truth, pred, labels=base.LABELS).tolist())}
                fold.append(row)
                part = df.iloc[test][base.META].copy()
                for key in ['mode', 'protocol', 'dataset', 'model', 'condition', 'train_session_set',
                            'initialization_session', 'test_session', 'ema_alpha', 'ema_alpha_fixed']:
                    part[key] = row[key]
                part['chronological_index'] = np.arange(1, len(test)+1)
                part['history_count_before_prediction'] = history
                part['y_true'], part['y_pred'] = truth, pred
                part['cumulative_accuracy'] = np.cumsum(truth == pred) / np.arange(1, len(test)+1)
                predictions.append(part)
            anchors = {row['condition']: row['accuracy'] for row in fold}
            for row in fold:
                row['gain_vs_zero_pp'] = 100 * (row['accuracy'] - anchors['zero_cal'])
                row['gain_vs_fixed_pp'] = 100 * (row['accuracy'] - anchors['fixed_session1']) if 'fixed_session1' in anchors else np.nan
            scores.extend(fold)
    per = pd.DataFrame(scores)
    pred = pd.concat(predictions, ignore_index=True)
    expected = sum(len(test) for _, _, _, test in plans) * len(datasets) * len(conditions)
    base.check(len(pred) == expected and not pred.duplicated(['dataset', 'condition'] + base.KEY).any(), 'Incomplete/duplicate predictions')
    summary = []
    for (name, condition), block in per.groupby(['dataset', 'condition'], sort=False):
        row = {'mode': mode, 'protocol': cs.PROTOCOL, 'dataset': name, 'condition': condition,
               'train_session_set': '1,2', 'initialization_session': int(block.initialization_session.iloc[0]),
               'test_session': 3, 'ema_alpha': alpha, 'ema_alpha_fixed': True,
               'n_subjects': len(block), 'n_test_windows': int(block.n_test_windows.sum())}
        for metric in METRICS:
            row[f'mean_{metric}'] = block[metric].mean()
            row[f'std_{metric}'] = block[metric].std(ddof=1)
            row[f'n_valid_{metric}'] = int(block[metric].notna().sum())
        summary.append(row)
    return per, pd.DataFrame(summary), pred, split_manifest(df, plans, conditions, mode)


def save_results(results, output, figures):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for frame, suffix in zip(results, ['per_subject', 'summary', 'predictions', 'split_manifest']):
        frame.to_csv(output / f'streaming_{suffix}.csv', index=False)
    summary = results[1]
    datasets = summary.dataset.unique().tolist()
    conditions = [c for c in CONDITIONS if c in set(summary.condition)]
    mode = summary['mode'].iloc[0].upper()
    for metric, label in [('accuracy', 'Accuracy'), ('macro_f1', 'Macro F1')]:
        fig, ax = plt.subplots(figsize=(11, 5))
        width = .8 / len(conditions)
        for i, condition in enumerate(conditions):
            block = summary[summary.condition == condition].set_index('dataset').loc[datasets]
            ax.bar(np.arange(len(datasets)) + (i - (len(conditions)-1)/2)*width,
                   block[f'mean_{metric}'], width, yerr=block[f'std_{metric}'], capsize=3, label=condition)
        ax.set_xticks(np.arange(len(datasets)), datasets)
        ax.set(ylabel=f'{label} (subject mean +/- sample SD)', ylim=(0, 1), title=f'{mode}: primary S3 prequential evaluation')
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f'streaming_{metric}.png', dpi=150)
        plt.close(fig)
    curve = cumulative_curve(results[2])
    curve.to_csv(output / 'streaming_cumulative_curve.csv', index=False)
    fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 5), squeeze=False)
    for ax, name in zip(axes[0], datasets):
        for condition in conditions:
            block = curve[(curve.dataset == name) & (curve.condition == condition)].sort_values('chronological_index')
            ax.plot(block.chronological_index, block.mean_cumulative_accuracy, label=condition)
        ax.set(title=name, xlabel='Chronological retained 60-s window index', ylabel='Mean subject cumulative accuracy', ylim=(0, 1))
        ax.legend()
    fig.suptitle(f'{mode}: primary S3; equal weight among subjects available at each index')
    fig.tight_layout()
    fig.savefig(figures / 'cumulative_accuracy_vs_time.png', dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--pilot', action='store_true')
    modes.add_argument('--require-all-subjects', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--datasets', nargs='+', choices=list(base.PARAM_SOURCE), default=list(base.PARAM_SOURCE))
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument('--ema-alpha', type=float, default=DEFAULT_ALPHA, help='Pre-specified fixed alpha; never tuned on test subjects')
    args = parser.parse_args(argv)
    try:
        validate_alpha(args.ema_alpha)
    except ValueError as error:
        parser.error(str(error))
    mode = 'pilot' if args.pilot else 'final'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = STAGE / 'output' / ('audit' if args.audit_only else mode) / stamp
    output.mkdir(parents=True)
    report = {'status': 'running', 'mode': mode, 'protocol': cs.PROTOCOL, 'audit_only': args.audit_only,
              'datasets': list(dict.fromkeys(args.datasets)), 'conditions': canonical_conditions(args.conditions),
              'ema_alpha': args.ema_alpha, 'ema_alpha_selection': 'pre-specified fixed, not tuned',
              'train_sessions': [1, 2], 'initialization_session': 1, 'test_session': 3,
              'held_out_session2': 'excluded', 'stream_order': ['trial', 'window_id'],
              'update_order': 'normalize with stats_(t-1) -> predict -> update raw x_t',
              'initialization': 'all held-out S1 windows, batch population mean/variance',
              'running_statistics': 'Welford; variance=M2/count; ddof=0',
              'ema_statistics': 'delta=x-mean_old; mean+=alpha*delta; variance=(1-alpha)*(variance+alpha*delta**2)',
              'ddof': 0, 'std_floor': cal.EPS, 'std_replacement': 1.0,
              'training_transform': 'raw -> train StandardScaler -> selector -> SVM',
              'zero_cal_transform': 'raw target -> train StandardScaler -> fitted selector -> fitted SVM',
              'adapted_transform': 'raw target -> target z-score -> fitted selector -> fitted SVM; bypass train scaler',
              'parameter_mapping': base.PARAM_SOURCE,
              'limitations': 'Offline SEED feature replay; causal normalization only, no classifier updates. Historical tuning on same cohort, not independent/nested selection.'}
    try:
        report.update(script_sha256=base.sha(__file__), cross_session_script_sha256=base.sha(CROSS_SESSION),
                      calibration_script_sha256=base.sha(cs.CALIBRATION), baseline_script_sha256=base.sha(cal.BASELINE),
                      config_sha256=base.sha(base.cfg.__file__), parameter_sha256=base.sha(base.TUNING),
                      parameters=base.fixed_parameters(),
                      feature_sets={k: v for k, v in base.feature_sets().items() if k in report['datasets']})
        df, audit = base.collect()
        audit['session_coverage'] = cs.session_coverage(df)
        for pilot, label in [(True, 'pilot'), (False, 'final')]:
            try:
                cs.eligibility(df, audit, pilot)
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
        cs.eligibility(df, audit, args.pilot)
        cal.write_json(output / 'run_report.json', report)
        results = evaluate(df, report['parameters'], mode, report['datasets'], report['conditions'], args.ema_alpha, audit)
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
