"""Synthetic model tests only; real inputs are inspected exclusively in audit mode."""
import json
import pickle
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_cross_session as cs


def synthetic_data():
    rng = np.random.default_rng(64)
    rows = []
    for subject in [1, 2, 3]:
        for session in [1, 2, 3]:
            for trial in [1, 2, 3]:
                # Unequal subject test counts also exercise subject-weighted summaries.
                for window in range(1, subject + 2):
                    raw = cs.base.cfg.TRIAL_LABELS[trial - 1]
                    label = cs.base.cfg.LABEL_MAP[raw]
                    row = dict(zip(cs.base.feature_sets()['Fusion'], rng.normal(3*subject + label, 1, 970)))
                    row.update(subject_id=f'S{subject:02d}', subject_num=subject,
                               session=int(cs.base.cfg.SUBJECT_SESSIONS[subject][session-1]),
                               session_index=session, trial=trial, window_id=window,
                               start_sample=(window-1)*6000, stop_sample=window*6000,
                               start_seconds=(window-1)*60, stop_seconds=window*60,
                               **{'class': label, 'class_label': cs.base.cfg.EMOTION_MAP[raw]})
                    rows.append(row)
    return pd.DataFrame(rows).sample(frac=1, random_state=6).reset_index(drop=True)


class CrossSessionTests(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_data()
        self.params = cs.base.fixed_parameters()

    def test_exact_splits_and_manifest(self):
        plans = cs.checked_splits(self.df)
        manifest = cs.split_manifest(self.df, plans, cs.CONDITIONS, 'pilot')
        self.assertNotIn('class', manifest)
        self.assertNotIn('class_label', manifest)
        self.assertEqual(len(manifest), len(self.df)*3*2)
        for subject, train, chosen, test in plans:
            held = self.df.subject_num == subject
            np.testing.assert_array_equal(train, np.flatnonzero(~held & self.df.session_index.isin([1, 2])))
            self.assertEqual(set(chosen), set(np.flatnonzero(held & (self.df.session_index == 1))))
            np.testing.assert_array_equal(test, np.flatnonzero(held & (self.df.session_index == 3)))
            self.assertFalse(set(self.df.iloc[train].subject_num) & {subject})
            excluded = set(np.flatnonzero(held & (self.df.session_index == 2)))
            self.assertFalse(excluded & (set(train) | set(chosen) | set(test)))
            block = manifest[(manifest.held_out_subject == subject) & (manifest.subject_num == subject)]
            self.assertTrue((block[block.session_index == 2].role == 'excluded_heldout_session2').all())
            self.assertTrue((block[block.session_index == 3].role == 'test').all())
        # Input DataFrame index is not mistaken for a positional index.
        shifted = self.df.copy()
        shifted.index += 1000
        for a, b in zip(plans, cs.checked_splits(shifted)):
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)

    def test_end_to_end_scaling_fit_state_and_zero_equivalence(self):
        original = cs.cal.predict_target
        previous = {}
        calls = []
        def guarded(pipe, X, chosen, test, condition):
            steps = [pipe[k] for k in ['scaler', 'select', 'clf']]
            state = [pickle.dumps(step.__dict__) for step in steps]
            if condition == '0':
                previous['steps'], previous['state'] = steps, state
                expected = pipe.predict(X[test])
            else:
                for a, b in zip(steps, previous['steps']):
                    self.assertIs(a, b)
                self.assertEqual(state, previous['state'])
                z, mu, sigma = cs.cal.transform_target(X, chosen, test)
                np.testing.assert_allclose(mu, X[chosen].mean(axis=0))
                self.assertTrue(np.isfinite(z).all())
                self.assertTrue(np.isfinite(sigma).all())
                expected = pipe['clf'].predict(pipe['select'].transform(z))
            with ExitStack() as stack:
                for step in [pipe] + steps:
                    stack.enter_context(patch.object(step, 'fit', side_effect=AssertionError('target refit')))
                if condition != '0':
                    stack.enter_context(patch.object(pipe['scaler'], 'transform', side_effect=AssertionError('double scaling')))
                    stack.enter_context(patch.object(pipe, 'predict', side_effect=AssertionError('pipeline on calibrated target')))
                result = original(pipe, X, chosen, test, condition)
            np.testing.assert_array_equal(result[0], expected)
            self.assertEqual(state, [pickle.dumps(step.__dict__) for step in steps])
            if condition == '0':
                self.assertIsNone(result[1])
                self.assertIsNone(result[2])
            calls.append(condition)
            return result
        with patch.object(cs.cal, 'predict_target', side_effect=guarded):
            results = cs.evaluate(self.df, self.params, 'pilot')
        self.assertEqual(len(calls), 18)
        per, summary, pred, _ = results
        self.assertEqual(dict(zip(per.dataset, per.n_features)), {'Spectral': 620, 'Connectivity': 350, 'Fusion': 970})
        self.assertTrue(np.isfinite(pred.y_pred).all())
        self.assertTrue(pred.y_pred.isin(cs.base.LABELS).all())
        self.assertEqual(len(pred), (self.df.session_index == 3).sum()*6)
        for name in cs.base.feature_sets():
            for subject, train, _, test in cs.checked_splits(self.df):
                columns = cs.base.feature_sets()[name]
                pipe = cs.base.build_pipeline(len(columns), self.params[name])
                pipe.fit(self.df.iloc[train][columns].to_numpy(), self.df.iloc[train]['class'])
                block = pred[(pred.dataset == name) & (pred.subject_num == subject)]
                zero = block[block.condition == 'zero_cal']
                calibrated = block[block.condition == 'session1_cal']
                np.testing.assert_array_equal(zero.y_pred, pipe.predict(self.df.iloc[test][columns].to_numpy()))
                pd.testing.assert_frame_equal(zero[cs.base.KEY].reset_index(drop=True), calibrated[cs.base.KEY].reset_index(drop=True))
                score = per[(per.dataset == name) & (per.subject_num == subject)].set_index('condition')
                self.assertAlmostEqual(score.loc['session1_cal', 'gain_pp'],
                                       100*(score.loc['session1_cal', 'accuracy'] - score.loc['zero_cal', 'accuracy']))
                self.assertEqual(score.loc['zero_cal', 'gain_pp'], 0)
        for _, row in summary.iterrows():
            block = per[(per.dataset == row.dataset) & (per.condition == row.condition)]
            for metric in cs.METRICS:
                self.assertAlmostEqual(row[f'mean_{metric}'], block[metric].mean())
                self.assertAlmostEqual(row[f'std_{metric}'], block[metric].std(ddof=1))
        with tempfile.TemporaryDirectory(dir=cs.STAGE / 'output') as tmp:
            cs.save_results(results, Path(tmp)/'output', Path(tmp)/'figures')
            self.assertEqual(len(list((Path(tmp)/'output').glob('*.csv'))), 4)
            self.assertEqual(len(list((Path(tmp)/'figures').glob('*.png'))), 2)

    def test_heldout_session2_changes_cannot_affect_its_fold(self):
        modified = self.df.copy()
        mask = (modified.subject_num == 1) & (modified.session_index == 2)
        modified.loc[mask, cs.base.feature_sets()['Fusion']] = 1e9
        modified.loc[mask, 'class'] = (modified.loc[mask, 'class'] + 1) % 3
        modified.loc[mask, 'class_label'] = 'ignored'
        before = cs.evaluate(self.df, self.params, 'pilot')
        after = cs.evaluate(modified, self.params, 'pilot')
        # S1's Session 2 legitimately trains OTHER held-out folds; compare S1's fold.
        for i in [0, 2]:
            pd.testing.assert_frame_equal(before[i][before[i].subject_num == 1].reset_index(drop=True),
                                          after[i][after[i].subject_num == 1].reset_index(drop=True))

    def test_exact_training_rows_and_single_fit_per_fold(self):
        build = cs.base.build_pipeline
        fitted = []
        with ExitStack() as stack:
            def tracked(n_features, params):
                pipe = build(n_features, params)
                spy = stack.enter_context(patch.object(pipe, 'fit', wraps=pipe.fit))
                fitted.append((pipe, spy))
                return pipe
            stack.enter_context(patch.object(cs.base, 'build_pipeline', side_effect=tracked))
            cs.evaluate(self.df, self.params, 'pilot')
            expected = [(columns, train) for columns in cs.base.feature_sets().values()
                        for _, train, _, _ in cs.checked_splits(self.df)]
            self.assertEqual(len(fitted), len(expected))
            for (pipe, spy), (columns, train) in zip(fitted, expected):
                spy.assert_called_once()
                X = self.df.iloc[train][columns].to_numpy()
                np.testing.assert_array_equal(spy.call_args.args[0], X)
                np.testing.assert_array_equal(spy.call_args.args[1], self.df.iloc[train]['class'])
                np.testing.assert_allclose(pipe['scaler'].mean_, X.mean(axis=0))

    def test_complete_session_coverage_and_missing_trial(self):
        # Coverage-only fixture: actual full-window integrity is inherited from collect.
        rows = []
        for subject, dates in cs.base.cfg.SUBJECT_SESSIONS.items():
            for index, date in enumerate(dates, 1):
                for trial, label in enumerate(cs.base.cfg.TRIAL_LABELS, 1):
                    rows.append({'subject_num': subject, 'session': int(date), 'session_index': index,
                                 'trial': trial, 'window_id': 1, 'class': cs.base.cfg.LABEL_MAP[label]})
        data = pd.DataFrame(rows)
        audit = {'complete_subjects': list(cs.base.cfg.SUBJECT_SESSIONS)}
        self.assertEqual(len(cs.eligibility(data, audit, False)), 15)
        missing = data[~((data.subject_num == 15) & (data.session_index == 3) & (data.trial == 15))]
        with self.assertRaisesRegex(ValueError, 'every expected trial'):
            cs.eligibility(missing, audit, False)

    def test_calibration_labels_do_not_change_stats_or_predictions(self):
        modified = self.df.copy()
        mask = (modified.subject_num == 1) & (modified.session_index == 1)
        modified.loc[mask, 'class'] = (modified.loc[mask, 'class'] + 1) % 3
        modified.loc[mask, 'class_label'] = 'unused'
        original = cs.cal.predict_target
        stats = []
        def capture(*args):
            result = original(*args)
            stats.append(result[1:])
            return result
        with patch.object(cs.cal, 'predict_target', side_effect=capture):
            before = cs.evaluate(self.df, self.params, 'pilot', ['Spectral'])
            after = cs.evaluate(modified, self.params, 'pilot', ['Spectral'])
        # First fold is held-out subject 1; second call is calibrated.
        for a, b in zip(stats[1], stats[7]):
            np.testing.assert_array_equal(a, b)
        for i in [0, 2]:
            pd.testing.assert_frame_equal(before[i][before[i].subject_num == 1].reset_index(drop=True),
                                          after[i][after[i].subject_num == 1].reset_index(drop=True))

    def test_missing_sessions_classes_and_duplicates_rejected_before_fit(self):
        cases = [self.df[~((self.df.subject_num == 3) & (self.df.session_index == index))] for index in [1, 3]]
        cases += [self.df[self.df.subject_num == 1], pd.concat([self.df, self.df.iloc[:1]])]
        no_class = self.df.copy()
        no_class.loc[no_class.session_index.isin([1, 2]), 'class'] = 0
        cases.append(no_class)
        with patch.object(cs.base, 'build_pipeline', side_effect=AssertionError('fit before validation')):
            for data in cases:
                with self.subTest(rows=len(data)), self.assertRaises(ValueError):
                    cs.evaluate(data, self.params, 'pilot')

    def test_final_coverage_gate_and_session_completeness(self):
        with self.assertRaisesRegex(ValueError, '15 COMPLETE'):
            cs.eligibility(self.df, {'complete_subjects': []}, False)
        # Even a purported complete-subject audit cannot hide missing session trials.
        with self.assertRaisesRegex(ValueError, 'every expected trial'):
            cs.eligibility(self.df, {'complete_subjects': list(cs.base.cfg.SUBJECT_SESSIONS)}, False)
        self.assertEqual(len(cs.eligibility(self.df, {'complete_subjects': []}, True)), 3)
        with self.assertRaisesRegex(ValueError, 'validated ingestion'):
            cs.evaluate(self.df, self.params, 'final')

    def test_schema_feature_order_and_conditions(self):
        self.assertEqual({k: len(v) for k, v in cs.base.validate_features(self.df).items()},
                         {'Spectral': 620, 'Connectivity': 350, 'Fusion': 970})
        before = cs.evaluate(self.df, self.params, 'pilot', ['Spectral'], ['session1_cal'])
        after = cs.evaluate(self.df[self.df.columns[::-1]], self.params, 'pilot', ['Spectral'])
        pd.testing.assert_frame_equal(before[2], after[2])
        self.assertEqual(cs.canonical_conditions(['session1_cal']), cs.CONDITIONS)
        self.assertEqual(cs.canonical_conditions(['zero_cal']), ['zero_cal'])
        zero_only = cs.evaluate(self.df, self.params, 'pilot', ['Spectral'], ['zero_cal'])
        self.assertEqual(set(zero_only[0].condition), {'zero_cal'})
        self.assertTrue((zero_only[0].n_calibration_windows == 0).all())
        bad = self.df.copy()
        bad.iloc[0, bad.columns.get_loc(cs.base.feature_sets()['Spectral'][0])] = np.inf
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            cs.evaluate(bad, self.params, 'pilot')

    def test_audit_only_never_evaluates(self):
        with tempfile.TemporaryDirectory(dir=cs.STAGE / 'output') as tmp:
            with (patch.object(cs, 'STAGE', Path(tmp)),
                  patch.object(cs, 'evaluate', side_effect=AssertionError('real experiment')),
                  patch.object(cs.base, 'build_pipeline', side_effect=AssertionError('real fit'))):
                self.assertEqual(cs.main(['--audit-only']), 0)
            reports = list(Path(tmp).glob('output/audit/*/run_report.json'))
            self.assertEqual(len(reports), 1)
            self.assertEqual(json.loads(reports[0].read_text())['status'], 'audit_only')
            self.assertTrue((reports[0].parent/'coverage_report.json').exists())
            self.assertFalse(list(Path(tmp).rglob('*.csv')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
