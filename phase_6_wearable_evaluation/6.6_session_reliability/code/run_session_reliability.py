"""Raw-feature test-retest reliability: matched targets measured in three SEED sessions."""
import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

STAGE = Path(__file__).resolve().parents[1]
BASELINE = STAGE.parent / '6.2_windowed_baseline/code/run_windowed_loso.py'
spec = importlib.util.spec_from_file_location('baseline62_reliability', BASELINE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
LEGACY = base.ROOT / 'phase_5_loso/5.5_icc_analysis/code/5.5_icc_analysis.ipynb'
UNIT = ['subject_num', 'trial', 'window_id']
SESSIONS = [1, 2, 3]
ICC_NAMES = ['ICC2_1', 'ICC3_1']
PRIMARY = 'matched_window_reliability'
DENOM_RTOL = 64 * np.finfo(float).eps


def feature_metadata(name):
    if name.startswith('spec_'):
        _, method, band, _ = name.split('_', 3)
        return {'feature_family': 'Spectral', 'method': 'PSD_power' if method == 'power' else 'Differential_entropy',
                'band': band, 'dataset_membership': 'Spectral;Fusion'}
    if name.startswith('gc_roi_'):
        return {'feature_family': 'Connectivity', 'method': 'GC_ROI', 'band': 'not_applicable',
                'dataset_membership': 'Connectivity;Fusion'}
    base.check(name.startswith('pdc_'), 'Unknown feature family')
    return {'feature_family': 'Connectivity', 'method': 'PDC_ROI', 'band': name.split('_')[1],
            'dataset_membership': 'Connectivity;Fusion'}


def match_units(df):
    """Metadata-only matching. Positions are local to df.reset_index(drop=True)."""
    meta = df[UNIT + ['session_index']].reset_index(drop=True)
    base.check(not meta.isna().any().any(), 'Missing matching metadata')
    base.check(meta.session_index.isin(SESSIONS).all(), 'Unknown session index')
    base.check(not meta.duplicated(UNIT + ['session_index']).any(), 'Duplicate unit/session')
    meta['row_position'] = np.arange(len(meta))
    wide = meta.pivot(index=UNIT, columns='session_index', values='row_position').reindex(columns=SESSIONS)
    wide = wide.sort_index()
    present = wide.notna()
    complete = present.all(axis=1)
    manifest = wide.index.to_frame(index=False)
    manifest['subject_id'] = manifest.subject_num.map(lambda x: f'S{int(x):02d}')
    for session in SESSIONS:
        manifest[f'present_s{session}'] = present[session].to_numpy()
    manifest['n_sessions_present'] = present.sum(axis=1).to_numpy()
    manifest['included_primary'] = complete.to_numpy()
    manifest['missing_sessions'] = [','.join(str(s) for s in SESSIONS if not row[s]) for _, row in present.iterrows()]
    manifest['status'] = np.where(complete, 'complete_3_session_unit', 'excluded_missing_session')
    positions = wide.loc[complete].to_numpy(dtype=int)
    return manifest, positions


def matching_coverage(df, manifest):
    subjects, trials = [], []
    for subject, dates in base.cfg.SUBJECT_SESSIONS.items():
        block = df[df.subject_num == subject]
        units = manifest[manifest.subject_num == subject]
        counts = []
        for session in SESSIONS:
            rows = block[block.session_index == session]
            counts.append({'session_index': session, 'session': int(dates[session-1]), 'windows': len(rows),
                           'passed_trials': int(rows.trial.nunique()),
                           'missing_trials': sorted(set(range(1, 16)) - set(rows.trial))})
        subjects.append({'subject_num': subject, 'available_sessions': sorted(int(s) for s in block.session_index.unique()),
                         'missing_sessions': sorted(set(SESSIONS) - set(block.session_index)),
                         'n_union_units': len(units), 'n_matched_units': int(units.included_primary.sum()),
                         'n_unmatched_units': int((~units.included_primary).sum()),
                         'n_unmatched_window_rows': int(units.loc[~units.included_primary, 'n_sessions_present'].sum()),
                         'session_coverage': counts})
        for trial in range(1, 16):
            trial_units = units[units.trial == trial]
            if len(trial_units) == 0 or not trial_units.included_primary.all():
                trials.append({'subject_num': subject, 'trial': trial,
                               'matched_window_ids': trial_units.loc[trial_units.included_primary, 'window_id'].astype(int).tolist(),
                               'unmatched_window_ids': trial_units.loc[~trial_units.included_primary, 'window_id'].astype(int).tolist(),
                               'missing_sessions_for_entire_trial': [s for s in SESSIONS if not trial_units[f'present_s{s}'].any()]})
    contributing = sorted(manifest.loc[manifest.included_primary, 'subject_num'].astype(int).unique().tolist())
    return {'unit': UNIT, 'repeated_measurements': SESSIONS, 'n_complete_3_session_units': int(manifest.included_primary.sum()),
            'n_unmatched_units': int((~manifest.included_primary).sum()), 'contributing_subjects': contributing,
            'available_subjects_without_matched_units': sorted(set(int(s) for s in df.subject_num.unique()) - set(contributing)),
            'per_subject': subjects, 'unmatched_or_missing_trials': trials}


def eligibility(df, audit, manifest, pilot):
    # Reuse coverage gate only; never call LOSO/class-availability checks or model code.
    base.eligibility(df, audit, pilot)
    counts = df.groupby('subject_num').session_index.nunique()
    base.check((counts >= 2).all(), 'Pilot requires at least two sessions per available subject')
    complete = manifest[manifest.included_primary]
    base.check(len(complete) >= 2 and complete.subject_num.nunique() >= 2,
               'Primary reliability requires complete three-session units from at least two subjects; no pairwise downgrade')
    if not pilot:
        base.check(set(complete.subject_num) == set(base.cfg.SUBJECT_SESSIONS), 'Final requires matched units for all 15 subjects')
        for subject in base.cfg.SUBJECT_SESSIONS:
            for session in SESSIONS:
                trials = set(df[(df.subject_num == subject) & (df.session_index == session)].trial)
                base.check(trials == set(range(1, 16)), 'Final requires 15 complete trials per subject/session')
            base.check(set(complete[complete.subject_num == subject].trial) == set(range(1, 16)),
                       'Final requires a three-session matched window for every subject/trial')


def icc_anova(values):
    """Vectorized explicit two-way ANOVA for [targets, 3 sessions, features]. No clipping."""
    X = np.asarray(values, dtype=float)
    if X.ndim == 2:
        X = X[:, :, None]
    base.check(X.ndim == 3 and X.shape[0] >= 2 and X.shape[1] == 3 and X.shape[2] > 0,
               'ICC requires >=2 targets and exactly three sessions')
    base.check(np.isfinite(X).all(), 'Non-finite/missing ICC values')
    n, k, _ = X.shape
    # Grand-mean centering is ANOVA arithmetic, not a subject/session normalization.
    centered = X - X.mean(axis=(0, 1))
    rows = centered.mean(axis=1)
    columns = centered.mean(axis=0)
    MSR = k * np.sum(rows**2, axis=0) / (n-1)
    MSC = n * np.sum(columns**2, axis=0) / (k-1)
    residual = centered - rows[:, None, :] - columns[None, :, :]
    MSE = np.sum(residual**2, axis=(0, 1)) / ((n-1)*(k-1))
    base.check(np.isfinite([MSR, MSC, MSE]).all(), 'Non-finite ANOVA mean squares')
    denominators = [MSR + (k-1)*MSE + k*(MSC-MSE)/n, MSR + (k-1)*MSE]
    tolerance = DENOM_RTOL * np.maximum.reduce([MSR, MSC, MSE])
    result = {'MSR': MSR, 'MSC': MSC, 'MSE': MSE}
    for name, denominator in zip(ICC_NAMES, denominators):
        valid = np.isfinite(denominator) & (denominator > tolerance)
        estimate = np.full_like(MSR, np.nan)
        np.divide(MSR-MSE, denominator, out=estimate, where=valid)
        valid &= np.isfinite(estimate)
        estimate[~valid] = np.nan
        result[name] = estimate
        result[f'{name}_status'] = np.where(valid, 'ok', 'undefined_denominator')
    return result


def correlation_columns(a, b):
    a, b = a-a.mean(axis=0), b-b.mean(axis=0)
    denominator = np.sqrt(np.sum(a*a, axis=0)) * np.sqrt(np.sum(b*b, axis=0))
    r = np.full(a.shape[1], np.nan)
    np.divide(np.sum(a*b, axis=0), denominator, out=r, where=np.isfinite(denominator) & (denominator > 0))
    r[~np.isfinite(r)] = np.nan
    return np.clip(r, -1, 1)  # Roundoff guard for correlations only, NEVER for ICC.


def pairwise_associations(cube):
    result = {}
    for left, right in [(0, 1), (0, 2), (1, 2)]:
        a, b = cube[:, left, :], cube[:, right, :]
        suffix = f'S{left+1}_S{right+1}'
        result[f'Pearson_{suffix}'] = correlation_columns(a, b)
        result[f'Spearman_{suffix}'] = correlation_columns(rankdata(a, axis=0, method='average'),
                                                          rankdata(b, axis=0, method='average'))
    return result


def draw_subject_indices(groups, rng):
    subjects = np.unique(groups)
    sampled = rng.choice(subjects, size=len(subjects), replace=True)
    # A selected subject contributes ALL its units, repeated if drawn more than once.
    indices = np.concatenate([np.flatnonzero(groups == subject) for subject in sampled])
    return sampled, indices


def bootstrap_ci(cube, groups, replicates=0, seed=42, observer=None):
    base.check(isinstance(replicates, (int, np.integer)) and replicates >= 0, 'Invalid bootstrap count')
    base.check(isinstance(seed, (int, np.integer)) and seed >= 0, 'Invalid seed')
    groups = np.asarray(groups)
    base.check(len(groups) == len(cube) and len(np.unique(groups)) >= 2, 'Bootstrap requires >=2 subject clusters')
    rng = np.random.default_rng(seed)
    samples = {name: np.full((replicates, cube.shape[2]), np.nan) for name in ICC_NAMES}
    for replicate in range(replicates):
        drawn, positions = draw_subject_indices(groups, rng)
        if observer is not None:
            observer(replicate, drawn.copy(), positions.copy())
        estimates = icc_anova(cube[positions])
        for name in ICC_NAMES:
            samples[name][replicate] = estimates[name]
    result = {}
    for name in ICC_NAMES:
        low, high = np.full(cube.shape[2], np.nan), np.full(cube.shape[2], np.nan)
        n_valid = np.isfinite(samples[name]).sum(axis=0)
        for feature in range(cube.shape[2]):
            valid = samples[name][:, feature][np.isfinite(samples[name][:, feature])]
            if len(valid) >= 2:
                low[feature], high[feature] = np.percentile(valid, [2.5, 97.5])
        result.update({f'{name}_ci_low': low, f'{name}_ci_high': high, f'{name}_bootstrap_valid': n_valid,
                       f'{name}_ci_status': np.where(n_valid >= 2, 'percentile_95',
                                                    'disabled' if replicates == 0 else 'insufficient_valid_replicates')})
    return result


def descriptive_bin(value):
    if not np.isfinite(value):
        return 'undefined'
    if value < .5:
        return 'poor'
    if value < .75:
        return 'moderate'
    if value < .9:
        return 'good'
    return 'excellent'


def feature_statistics(cube, groups, features, mode, replicates, seed, analysis=PRIMARY, n_matched=None):
    stats = {**icc_anova(cube), **pairwise_associations(cube), **bootstrap_ci(cube, groups, replicates, seed)}
    rows = []
    for i, feature in enumerate(features):
        row = {'mode': mode, 'analysis': analysis, 'normalization': 'raw', 'feature_name': feature,
               **feature_metadata(feature), 'n_targets': len(cube),
               'n_matched_units': len(cube) if n_matched is None else n_matched,
               'n_subjects': len(np.unique(groups)), 'n_sessions': 3, 'bootstrap_replicates': replicates,
               'bootstrap_seed': seed, 'bootstrap_unit': 'subject', **{k: v[i] for k, v in stats.items()}}
        for name in ICC_NAMES:
            row[f'{name}_bin'] = descriptive_bin(row[name])
        rows.append(row)
    return pd.DataFrame(rows)


def family_summary(per):
    rows = []
    groups = [('method_band', keys, block) for keys, block in per.groupby(['feature_family', 'method', 'band'], sort=False)]
    groups += [('spectral_method_all_bands', (family, method, 'all_spectral_bands'), block)
               for (family, method), block in per[per.feature_family == 'Spectral'].groupby(['feature_family', 'method'], sort=False)]
    for level, (family, method, band), block in groups:
        for metric in ICC_NAMES:
            valid = block.loc[np.isfinite(block[metric]), metric]
            row = {'mode': per['mode'].iloc[0], 'analysis': PRIMARY, 'summary_level': level,
                   'feature_family': family, 'method': method, 'band': band, 'icc_type': metric,
                   'n_features': len(block), 'n_defined': len(valid), 'n_undefined': len(block)-len(valid),
                   'median_ICC': valid.median(), 'mean_ICC': valid.mean(),
                   'q25_ICC': valid.quantile(.25), 'q75_ICC': valid.quantile(.75),
                   'IQR_ICC': valid.quantile(.75)-valid.quantile(.25)}
            for threshold, suffix in [(.5, '050'), (.75, '075'), (.9, '090')]:
                row[f'proportion_ge_{suffix}'] = float((valid >= threshold).mean()) if len(valid) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def pairwise_table(per):
    parts = []
    identity = ['mode', 'analysis', 'feature_name', 'feature_family', 'method', 'band', 'dataset_membership',
                'n_matched_units', 'n_subjects', 'n_sessions']
    for pair in ['S1_S2', 'S1_S3', 'S2_S3']:
        part = per[identity].copy()
        part['session_pair'] = pair
        part['Pearson_r'], part['Spearman_rho'] = per[f'Pearson_{pair}'], per[f'Spearman_{pair}']
        part['association_status'] = np.where(part.Pearson_r.notna() & part.Spearman_rho.notna(), 'ok', 'undefined_constant_session')
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def analyze(df, mode, datasets=None, replicates=0, seed=42, audit=None, subject_means=False):
    base.check(mode in ['pilot', 'final'], 'Invalid mode')
    df = df.reset_index(drop=True)
    sets = base.validate_features(df)
    datasets = list(sets) if datasets is None else list(datasets)
    base.check(bool(datasets) and len(set(datasets)) == len(datasets) and set(datasets) <= set(sets), 'Invalid datasets')
    manifest, positions = match_units(df)
    if mode == 'final':
        base.check(audit is not None, 'Final requires validated ingestion coverage')
    eligibility(df, audit if audit is not None else {'complete_subjects': []}, manifest, mode == 'pilot')
    wanted = {feature for dataset in datasets for feature in sets[dataset]}
    features = [feature for feature in sets['Fusion'] if feature in wanted]  # Unique union, no Fusion duplication.
    cube = df[features].to_numpy(dtype=float)[positions]
    groups = manifest.loc[manifest.included_primary, 'subject_num'].to_numpy()
    per = feature_statistics(cube, groups, features, mode, replicates, seed)
    result = {'reliability_per_feature': per, 'reliability_family_summary': family_summary(per),
              'reliability_pairwise': pairwise_table(per), 'reliability_matched_unit_manifest': manifest}
    if subject_means:
        subjects = np.unique(groups)
        means = np.stack([cube[groups == subject].mean(axis=0) for subject in subjects])
        result['subject_session_mean_reliability'] = feature_statistics(
            means, subjects, features, mode, replicates, seed, 'subject_session_mean_reliability', len(cube))
    return result


def save_results(results, output, figures):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        frame.to_csv(output / f'{name}.csv', index=False)
    per = results['reliability_per_feature']
    for metric, filename, title in [('ICC2_1', 'icc2_distribution', 'ICC(2,1): absolute agreement'),
                                    ('ICC3_1', 'icc3_distribution', 'ICC(3,1): consistency')]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for (method, band), block in per.groupby(['method', 'band'], sort=False):
            values = block[metric].dropna()
            if len(values):
                ax.hist(values, bins=20, histtype='step', label=f'{method}/{band}')
        ax.set(xlabel=title, ylabel='Feature count', title=f'{per["mode"].iloc[0].upper()}: raw matched-unit {title}')
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(figures / f'{filename}.png', dpi=150)
        plt.close(fig)
    summary = results['reliability_family_summary']
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    for ax, metric in zip(axes, ICC_NAMES):
        block = summary[(summary.icc_type == metric) & (summary.summary_level == 'method_band')]
        x = np.arange(len(block))
        ax.bar(x, block.median_ICC, color='steelblue')
        ax.vlines(x, block.q25_ICC, block.q75_ICC, color='black', label='Feature IQR (not CI)')
        ax.set_xticks(x, block.method + '/' + block.band, rotation=60, ha='right')
        ax.set(ylabel=f'Median {metric}', title='Absolute agreement' if metric == 'ICC2_1' else 'Consistency')
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures / 'icc_by_feature_family.png', dpi=150)
    plt.close(fig)
    pairwise = results['reliability_pairwise']
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    for ax, metric in zip(axes, ['Pearson_r', 'Spearman_rho']):
        grouped = pairwise.groupby(['method', 'band', 'session_pair'], sort=False)[metric].median().unstack('session_pair')
        for pair in ['S1_S2', 'S1_S3', 'S2_S3']:
            ax.plot(np.arange(len(grouped)), grouped[pair], marker='o', label=pair)
        ax.set_xticks(np.arange(len(grouped)), [f'{m}/{b}' for m, b in grouped.index], rotation=60, ha='right')
        ax.set(ylabel=f'Median {metric}', ylim=(-1.05, 1.05), title='Pairwise association/stability (not ICC)')
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures / 'pairwise_session_correlation.png', dpi=150)
    plt.close(fig)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--pilot', action='store_true')
    modes.add_argument('--require-all-subjects', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--bootstrap', type=int, default=0, help='Subject-cluster replicates; request 1000 for final CIs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--datasets', nargs='+', choices=list(base.feature_sets()), default=list(base.feature_sets()))
    parser.add_argument('--subject-session-means', action='store_true', help='Separate secondary ICC of matched-unit subject/session means')
    args = parser.parse_args(argv)
    if args.bootstrap < 0 or args.seed < 0:
        parser.error('Bootstrap and seed must be nonnegative')
    mode = 'pilot' if args.pilot else 'final'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = STAGE / 'output' / ('audit' if args.audit_only else mode) / stamp
    output.mkdir(parents=True)
    report = {'status': 'running', 'mode': mode, 'audit_only': args.audit_only, 'analysis': PRIMARY,
              'datasets': list(dict.fromkeys(args.datasets)), 'unit': UNIT, 'session_columns': SESSIONS,
              'normalization': 'raw extracted features; no subject-wise or train scaling',
              'bootstrap_requested': args.bootstrap, 'bootstrap_executed': 0, 'seed': args.seed,
              'bootstrap_method': 'subject-cluster resampling, all matched units per draw; percentile 95% CI',
              'subject_session_means': args.subject_session_means, 'denominator_relative_tolerance': DENOM_RTOL,
              'ICC2_1': '(MSR-MSE)/(MSR+(k-1)*MSE+k*(MSC-MSE)/n)',
              'ICC3_1': '(MSR-MSE)/(MSR+(k-1)*MSE)',
              'negative_ICC': 'retained', 'undefined': 'NaN/blank CSV plus explicit status',
              'legacy_distinction': 'Phase 5 grouped by subject only; not session test-retest reliability.',
              'limitations': 'Repeated-stimulus offline SEED context; targets nested in subjects; sessions are ordered occasions, not shared calendar dates.'}
    try:
        report.update(script_sha256=base.sha(__file__), baseline_script_sha256=base.sha(BASELINE),
                      config_sha256=base.sha(base.cfg.__file__), legacy_notebook_sha256=base.sha(LEGACY),
                      feature_sets={k: v for k, v in base.feature_sets().items() if k in report['datasets']})
        df, audit = base.collect()
        manifest, _ = match_units(df)
        audit['matching'] = matching_coverage(df, manifest)
        for pilot, label in [(True, 'pilot'), (False, 'final')]:
            try:
                eligibility(df, audit, manifest, pilot)
                audit[f'{label}_eligibility'] = {'eligible': True}
            except ValueError as error:
                audit[f'{label}_eligibility'] = {'eligible': False, 'reason': str(error)}
        write_json(output / 'coverage_report.json', audit)
        manifest.to_csv(output / 'reliability_matched_unit_manifest.csv', index=False)
        print(f'Available: {audit["available_subjects"]}; complete: {audit["complete_subjects"]}; windows: {len(df)}; matched units: {audit["matching"]["n_complete_3_session_units"]}')
        if args.audit_only:
            report['status'] = 'audit_only'
            write_json(output / 'run_report.json', report)
            print(f'No reliability estimates or bootstrap computed. Audit: {output}')
            return 0
        eligibility(df, audit, manifest, args.pilot)
        write_json(output / 'run_report.json', report)
        results = analyze(df, mode, report['datasets'], args.bootstrap, args.seed, audit, args.subject_session_means)
        save_results(results, output, STAGE / 'figures' / mode / stamp)
        report.update(status='passed', bootstrap_executed=args.bootstrap,
                      n_matched_units=audit['matching']['n_complete_3_session_units'],
                      n_contributing_subjects=len(audit['matching']['contributing_subjects']))
        write_json(output / 'run_report.json', report)
        print(f'Results: {output}')
        return 0
    except (ValueError, KeyError, OSError) as error:
        report.update(status='rejected', reason=str(error))
        write_json(output / 'run_report.json', report)
        print(f'Rejected: {error}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
