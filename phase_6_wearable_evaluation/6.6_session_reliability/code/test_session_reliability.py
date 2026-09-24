"""Synthetic measurement-stability checks; no real-data reliability or classification."""
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import run_session_reliability as rel


def synthetic_data():
    rng = np.random.default_rng(66)
    features = rel.base.feature_sets()['Fusion']
    rows = []
    for subject in [1, 2, 3]:
        for trial in [1, 2, 3]:
            for window in range(1, subject+2):
                target = rng.normal(subject+trial, 1, len(features))
                for session in [1, 2, 3]:
                    values = target + rng.normal(0, .3, len(features)) + .2*session
                    values[0] = 5.0  # Undefined constant feature must remain in outputs.
                    row = dict(zip(features, values))
                    row.update(subject_num=subject, subject_id=f'S{subject:02d}', trial=trial, window_id=window,
                               session_index=session, session=int(rel.base.cfg.SUBJECT_SESSIONS[subject][session-1]),
                               start_sample=(window-1)*6000, stop_sample=window*6000,
                               start_seconds=(window-1)*60, stop_seconds=window*60,
                               **{'class': 0, 'class_label': 'not_used'})
                    rows.append(row)
    return pd.DataFrame(rows).sample(frac=1, random_state=66).reset_index(drop=True)


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_data()

    def test_perfect_reliability_and_session_offset(self):
        targets = np.arange(1., 21.)
        perfect = np.repeat(targets[:, None], 3, axis=1)
        stats = rel.icc_anova(perfect)
        for name in rel.ICC_NAMES:
            np.testing.assert_allclose(stats[name], 1, atol=1e-12)
        shifted = perfect + [0, 10, 20]
        stats = rel.icc_anova(shifted)
        np.testing.assert_allclose(stats['ICC3_1'], 1, atol=1e-12)
        self.assertLess(stats['ICC2_1'][0], .5)
        self.assertGreater(stats['MSC'][0], 0)

    def test_random_and_negative_estimates_not_clipped(self):
        X = np.random.default_rng(10).normal(size=(3000, 3))
        for name in rel.ICC_NAMES:
            self.assertLess(abs(rel.icc_anova(X)[name][0]), .05)
        anti = np.array([[1., 0., -1.], [-1., 1., 0.], [0., -1., 1.]])
        result = rel.icc_anova(anti)
        for name in rel.ICC_NAMES:
            self.assertLess(result[name][0], 0)
            self.assertEqual(rel.descriptive_bin(result[name][0]), 'poor')

    def test_constant_feature_missing_nonfinite_and_shape_handling(self):
        stats = rel.icc_anova(np.full((5, 3), 7.))
        for name in rel.ICC_NAMES:
            self.assertTrue(np.isnan(stats[name][0]))
            self.assertEqual(stats[f'{name}_status'][0], 'undefined_denominator')
        for bad in [np.ones((1, 3)), np.ones((4, 2)), np.array([[1., np.nan, 2.], [2., 3., 4.]]),
                    np.array([[1., np.inf, 2.], [2., 3., 4.]])]:
            with self.assertRaises(ValueError):
                rel.icc_anova(bad)
        # Pure session offsets, no target differences: consistency is unidentified.
        result = rel.icc_anova(np.tile([1., 2., 3.], (5, 1)))
        np.testing.assert_allclose(result['ICC2_1'], 0, atol=1e-12)
        self.assertTrue(np.isnan(result['ICC3_1'][0]))
        self.assertEqual([rel.descriptive_bin(v) for v in [.49, .5, .75, .9, np.nan]],
                         ['poor', 'moderate', 'good', 'excellent', 'undefined'])

    def test_session_target_permutation_and_common_scale_invariance(self):
        X = np.random.default_rng(6).normal(size=(20, 3, 4))
        a = rel.icc_anova(X)
        for changed in [X[:, [2, 0, 1]], X[::-1], X*1e-8, X*5 + 20]:
            b = rel.icc_anova(changed)
            for name in rel.ICC_NAMES:
                np.testing.assert_allclose(a[name], b[name], atol=1e-12)

    def test_independent_least_squares_anova_reference(self):
        X = np.array([[1., 2., 3.], [2., 2., 4.], [5., 4., 6.], [8., 7., 9.]])
        n, k = X.shape
        y = X.ravel()
        intercept = np.ones((n*k, 1))
        row_dummies = np.repeat(np.eye(n)[:, 1:], k, axis=0)
        col_dummies = np.tile(np.eye(k)[:, 1:], (n, 1))
        def sse(design):
            beta = np.linalg.lstsq(design, y, rcond=None)[0]
            return np.sum((y-design@beta)**2)
        # Independent regression projection, not row/column-mean implementation.
        residual = sse(np.column_stack([intercept, row_dummies, col_dummies]))
        MSR = (sse(np.column_stack([intercept, col_dummies])) - residual)/(n-1)
        MSC = (sse(np.column_stack([intercept, row_dummies])) - residual)/(k-1)
        MSE = residual/((n-1)*(k-1))
        stats = rel.icc_anova(X)
        for name, value in [('MSR', MSR), ('MSC', MSC), ('MSE', MSE)]:
            np.testing.assert_allclose(stats[name], value, atol=1e-12)
        expected2 = (MSR-MSE)/(MSR+(k-1)*MSE+k*(MSC-MSE)/n)
        expected3 = (MSR-MSE)/(MSR+(k-1)*MSE)
        np.testing.assert_allclose(stats['ICC2_1'], expected2, atol=1e-12)
        np.testing.assert_allclose(stats['ICC3_1'], expected3, atol=1e-12)

    def test_matching_missing_units_and_no_cross_subject_or_trial_match(self):
        missing = (self.df.subject_num == 1) & (self.df.trial == 2) & (self.df.window_id == 1) & (self.df.session_index == 3)
        data = self.df[~missing].copy()
        manifest, positions = rel.match_units(data)
        excluded = manifest[~manifest.included_primary]
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[rel.UNIT].iloc[0].tolist(), [1, 2, 1])
        self.assertEqual(excluded.missing_sessions.iloc[0], '3')
        self.assertEqual(positions.shape, (26, 3))
        reset = data.reset_index(drop=True)
        for target in positions:
            block = reset.iloc[target]
            self.assertEqual(len(block[rel.UNIT].drop_duplicates()), 1)
            self.assertEqual(block.session_index.tolist(), [1, 2, 3])
        coverage = rel.matching_coverage(data, manifest)
        self.assertEqual(coverage['n_unmatched_units'], 1)
        self.assertEqual(coverage['per_subject'][0]['n_unmatched_window_rows'], 2)
        self.assertTrue(any(row['subject_num'] == 1 and row['trial'] == 2 and row['unmatched_window_ids'] == [1]
                            for row in coverage['unmatched_or_missing_trials']))
        # Different window counts: only the intersection is included.
        short = data[~((data.subject_num == 2) & (data.session_index == 2) & (data.window_id == 3))]
        other, _ = rel.match_units(short)
        self.assertEqual(int((~other.included_primary).sum()), 4)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            rel.match_units(pd.concat([self.df, self.df.iloc[:1]]))

    def test_no_labels_feature_order_independence_and_no_classifier_calls(self):
        changed = self.df.copy()
        changed['class'] = -999
        changed['class_label'] = 'unused'
        with ExitStack() as stack:
            for name in ['fixed_parameters', 'build_pipeline', 'evaluate', 'checked_splits']:
                stack.enter_context(patch.object(rel.base, name, side_effect=AssertionError('classifier code called')))
            original = rel.analyze(self.df, 'pilot')
            after = rel.analyze(changed[changed.columns[::-1]], 'pilot')
        for key in original:
            pd.testing.assert_frame_equal(original[key], after[key])
        per = original['reliability_per_feature']
        self.assertEqual(len(per), 970)  # Spectral + Connectivity + Fusion must not duplicate features.
        self.assertFalse(per.feature_name.duplicated().any())
        self.assertEqual(set(per.normalization), {'raw'})
        self.assertEqual(per.ICC2_1.isna().sum(), 1)
        self.assertEqual(set(per.n_matched_units), {27})
        self.assertEqual(set(per.n_subjects), {3})
        self.assertEqual(len(original['reliability_pairwise']), 2910)
        pdc = original['reliability_family_summary'].query("method == 'PDC_ROI'")
        self.assertEqual(set(pdc.band), {'delta', 'theta', 'alpha', 'beta', 'gamma', 'broadband'})
        self.assertTrue((pdc.n_features == 50).all())
        spectral = original['reliability_family_summary'].query("feature_family == 'Spectral' and summary_level == 'method_band'")
        self.assertTrue((spectral.n_features == 62).all())

    def test_pairwise_correlations_against_scipy_and_no_pairwise_extra_rows(self):
        cube = np.array([[[1, 5], [2, 5], [3, 5]], [[2, 5], [2, 5], [1, 5]],
                         [[3, 5], [4, 5], [2, 5]], [[4, 5], [3, 5], [4, 5]]], dtype=float)
        result = rel.pairwise_associations(cube)
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            pair = f'S{a+1}_S{b+1}'
            self.assertAlmostEqual(result[f'Pearson_{pair}'][0], pearsonr(cube[:, a, 0], cube[:, b, 0]).statistic)
            self.assertAlmostEqual(result[f'Spearman_{pair}'][0], spearmanr(cube[:, a, 0], cube[:, b, 0]).statistic)
            self.assertTrue(np.isnan(result[f'Pearson_{pair}'][1]))
        missing = self.df[~((self.df.subject_num == 1) & (self.df.session_index == 3) & (self.df.window_id == 1))]
        result = rel.analyze(missing, 'pilot', ['Spectral'])
        self.assertEqual(set(result['reliability_pairwise'].n_matched_units), {24})

    def test_bootstrap_subject_clusters_reproducibility_and_percentiles(self):
        rng = np.random.default_rng(8)
        cube = rng.normal(size=(9, 3, 2))
        groups = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3])
        draws = []
        a = rel.bootstrap_ci(cube, groups, 30, 42, observer=lambda *args: draws.append(args))
        b = rel.bootstrap_ci(cube, groups, 30, 42)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        expected = {name: [] for name in rel.ICC_NAMES}
        self.assertTrue(any(len(set(sampled)) < 3 for _, sampled, _ in draws))
        for _, sampled, indices in draws:
            expected_indices = np.concatenate([np.flatnonzero(groups == subject) for subject in sampled])
            np.testing.assert_array_equal(indices, expected_indices)
            for subject in np.unique(groups):
                self.assertEqual(np.sum(groups[indices] == subject), np.sum(sampled == subject)*np.sum(groups == subject))
            stats = rel.icc_anova(cube[indices])
            for name in rel.ICC_NAMES:
                expected[name].append(stats[name])
        for name in rel.ICC_NAMES:
            low, high = np.percentile(expected[name], [2.5, 97.5], axis=0)
            np.testing.assert_allclose(a[f'{name}_ci_low'], low)
            np.testing.assert_allclose(a[f'{name}_ci_high'], high)
            np.testing.assert_array_equal(a[f'{name}_bootstrap_valid'], [30, 30])
        disabled = rel.bootstrap_ci(cube, groups, 0)
        self.assertTrue(np.isnan(disabled['ICC2_1_ci_low']).all())
        self.assertTrue((disabled['ICC2_1_ci_status'] == 'disabled').all())
        undefined = rel.bootstrap_ci(np.ones((9, 3, 1)), groups, 5)
        self.assertTrue(np.isnan(undefined['ICC3_1_ci_high']).all())
        self.assertEqual(undefined['ICC3_1_bootstrap_valid'][0], 0)

    def test_family_summary_includes_negative_and_undefined(self):
        per = pd.DataFrame({'mode': ['pilot']*5, 'feature_family': ['Spectral']*5,
                            'method': ['PSD_power']*5, 'band': ['alpha']*5,
                            'ICC2_1': [-.2, .5, .8, .95, np.nan], 'ICC3_1': [-.2, .5, .8, .95, np.nan]})
        result = rel.family_summary(per)
        row = result.iloc[0]
        self.assertEqual(row.n_features, 5)
        self.assertEqual(row.n_defined, 4)
        self.assertEqual(row.n_undefined, 1)
        self.assertAlmostEqual(row.mean_ICC, np.mean([-.2, .5, .8, .95]))
        self.assertEqual(row.proportion_ge_050, .75)
        self.assertEqual(row.proportion_ge_075, .5)
        self.assertEqual(row.proportion_ge_090, .25)

    def test_pilot_missing_session_no_silent_downgrade_and_final_gate(self):
        only_two = self.df[self.df.session_index != 3]
        with self.assertRaisesRegex(ValueError, 'no pairwise downgrade'):
            rel.analyze(only_two, 'pilot')
        one_missing_s3 = self.df[~((self.df.subject_num == 3) & (self.df.session_index == 3))]
        manifest, _ = rel.match_units(one_missing_s3)
        audit = rel.matching_coverage(one_missing_s3, manifest)
        self.assertEqual(audit['available_subjects_without_matched_units'], [3])
        result = rel.analyze(one_missing_s3, 'pilot', ['Spectral'])
        self.assertEqual(set(result['reliability_per_feature'].n_subjects), {2})
        with self.assertRaisesRegex(ValueError, 'validated ingestion'):
            rel.analyze(self.df, 'final')
        with self.assertRaisesRegex(ValueError, '15 COMPLETE'):
            rel.analyze(self.df, 'final', audit={'complete_subjects': []})

    def test_secondary_means_and_all_outputs(self):
        result = rel.analyze(self.df, 'pilot', replicates=6, seed=42, subject_means=True)
        secondary = result['subject_session_mean_reliability']
        self.assertEqual(set(secondary.analysis), {'subject_session_mean_reliability'})
        self.assertEqual(set(secondary.n_targets), {3})
        self.assertEqual(set(secondary.n_matched_units), {27})
        feature = rel.base.feature_sets()['Fusion'][1]
        batch = self.df.groupby(['subject_num', 'session_index'])[feature].mean().unstack('session_index').to_numpy()
        expected = rel.icc_anova(batch)
        for metric in rel.ICC_NAMES:
            self.assertAlmostEqual(secondary.set_index('feature_name').loc[feature, metric], expected[metric][0])
        with tempfile.TemporaryDirectory(dir=rel.STAGE / 'output') as tmp:
            rel.save_results(result, Path(tmp)/'output', Path(tmp)/'figures')
            self.assertEqual(len(list((Path(tmp)/'output').glob('*.csv'))), 5)
            self.assertEqual({p.name for p in (Path(tmp)/'figures').glob('*.png')},
                             {'icc2_distribution.png', 'icc3_distribution.png', 'icc_by_feature_family.png', 'pairwise_session_correlation.png'})

    def test_complete_final_metadata_and_empty_coverage(self):
        # Coverage-only fixture: ingestion supplies full-window validation, not this fixture.
        rows = [{'subject_num': subject, 'trial': trial, 'window_id': 1, 'session_index': session}
                for subject in rel.base.cfg.SUBJECT_SESSIONS for trial in range(1, 16) for session in [1, 2, 3]]
        data = pd.DataFrame(rows)
        manifest, positions = rel.match_units(data)
        self.assertEqual(positions.shape, (225, 3))
        audit = {'complete_subjects': list(rel.base.cfg.SUBJECT_SESSIONS)}
        rel.eligibility(data, audit, manifest, False)
        missing = data[~((data.subject_num == 15) & (data.trial == 15) & (data.session_index == 3))]
        incomplete, _ = rel.match_units(missing)
        with self.assertRaisesRegex(ValueError, '15 complete trials'):
            rel.eligibility(missing, audit, incomplete, False)
        empty, positions = rel.match_units(self.df.iloc[:0])
        self.assertEqual(positions.shape, (0, 3))
        self.assertEqual(rel.matching_coverage(self.df.iloc[:0], empty)['n_complete_3_session_units'], 0)

    def test_audit_only_no_reliability_bootstrap_or_classifier(self):
        with tempfile.TemporaryDirectory(dir=rel.STAGE / 'output') as tmp:
            with ExitStack() as stack:
                stack.enter_context(patch.object(rel, 'STAGE', Path(tmp)))
                for obj, name in [(rel, 'analyze'), (rel, 'bootstrap_ci'), (rel.base, 'build_pipeline'), (rel.base, 'fixed_parameters')]:
                    stack.enter_context(patch.object(obj, name, side_effect=AssertionError('analysis during audit')))
                self.assertEqual(rel.main(['--audit-only', '--bootstrap', '1000']), 0)
            reports = list(Path(tmp).glob('output/audit/*/run_report.json'))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            self.assertEqual(report['status'], 'audit_only')
            self.assertEqual(report['bootstrap_executed'], 0)
            self.assertEqual(report['bootstrap_requested'], 1000)
            self.assertEqual({p.name for p in reports[0].parent.glob('*.csv')}, {'reliability_matched_unit_manifest.csv'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
