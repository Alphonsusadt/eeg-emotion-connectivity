"""Synthetic prequential tests; real data is only inspected in audit mode."""
import json
import pickle
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_streaming_normalization as sn


def synthetic_data():
    rng = np.random.default_rng(65)
    rows = []
    for subject in [1, 2, 3]:
        for session in [1, 2, 3]:
            for trial in [1, 2, 3]:
                for window in range(1, subject + 2):
                    raw = sn.base.cfg.TRIAL_LABELS[trial-1]
                    label = sn.base.cfg.LABEL_MAP[raw]
                    row = dict(zip(sn.base.feature_sets()['Fusion'], rng.normal(subject*2 + label, 1, 970)))
                    row.update(subject_num=subject, subject_id=f'S{subject:02d}', session_index=session,
                               session=int(sn.base.cfg.SUBJECT_SESSIONS[subject][session-1]), trial=trial,
                               window_id=window, start_sample=(window-1)*6000, stop_sample=window*6000,
                               start_seconds=(window-1)*60, stop_seconds=window*60,
                               **{'class': label, 'class_label': sn.base.cfg.EMOTION_MAP[raw]})
                    rows.append(row)
    return pd.DataFrame(rows).sample(frac=1, random_state=65).reset_index(drop=True)


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_data()
        self.params = sn.base.fixed_parameters()

    def small_model(self):
        train = np.arange(98., 104.).reshape(-1, 1)
        pipe = sn.base.build_pipeline(1, {'C': 100, 'gamma': 1})
        pipe.fit(train, [0, 0, 1, 1, 2, 2])
        return pipe

    def test_welford_matches_numpy_every_prefix_and_constant_memory(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(80, 5)) + 1e6
        state = sn.RunningStatistics(X[:1])
        for n in range(1, len(X)+1):
            np.testing.assert_allclose(state.mean, X[:n].mean(axis=0), atol=1e-8, rtol=0)
            np.testing.assert_allclose(state.M2/state.count, X[:n].var(axis=0, ddof=0), atol=1e-8, rtol=0)
            np.testing.assert_allclose(state.std(), sn.safe_std(X[:n].var(axis=0)), atol=1e-8, rtol=0)
            self.assertEqual(state.count, n)
            self.assertEqual(set(state.__dict__), {'count', 'mean', 'M2'})
            self.assertEqual(state.mean.shape, (5,))
            self.assertEqual(state.M2.shape, (5,))
            if n < len(X):
                state.update(X[n])
        batch_seed = sn.RunningStatistics(X[:12])
        for x in X[12:]:
            batch_seed.update(x)
        np.testing.assert_allclose(batch_seed.std(), X.std(axis=0), atol=1e-8, rtol=0)

    def test_ema_manual_recurrence(self):
        state = sn.EMAStatistics(np.array([[1., 2.], [3., 6.]]), .1)
        np.testing.assert_array_equal(state.mean, [2, 4])
        np.testing.assert_array_equal(state.variance, [1, 4])
        state.update([6, 8])
        np.testing.assert_allclose(state.mean, [2.4, 4.4])
        np.testing.assert_allclose(state.variance, [2.34, 5.04])
        state.update([0, 2])
        np.testing.assert_allclose(state.mean, [2.16, 4.16])
        np.testing.assert_allclose(state.variance, [2.6244, 5.0544])
        np.testing.assert_allclose(state.std(), np.sqrt([2.6244, 5.0544]))
        self.assertEqual(state.count, 4)

    def test_zero_variance_fallback_does_not_replace_stored_variance(self):
        for cls in [sn.RunningStatistics, sn.EMAStatistics]:
            state = cls(np.array([[20.]]))
            np.testing.assert_array_equal(state.std(), [1])
            state.update(np.array([22.]))
            expected = 1.0 if cls == sn.RunningStatistics else .36
            np.testing.assert_allclose(state.std(), [np.sqrt(expected)])
        np.testing.assert_array_equal(sn.safe_std(np.array([0., 1e-22, 1.])), [1, 1, 1])

    def test_strict_predict_before_update_prefix_statistics(self):
        pipe = self.small_model()
        initial = np.array([[20.], [22.]])
        stream = np.array([[21.], [24.], [20.], [22.]])
        for condition in ['streaming_running', 'streaming_ema']:
            mean, variance = initial.mean(axis=0), initial.var(axis=0)
            expected = []
            for index, x in enumerate(stream):
                if condition == 'streaming_running':
                    history = np.concatenate([initial, stream[:index]])
                    mean, variance = history.mean(axis=0), history.var(axis=0)
                std = sn.safe_std(variance)
                z = ((x-mean)/std).reshape(1, -1)
                expected.append((mean.copy(), std, z, pipe['clf'].predict(pipe['select'].transform(z))))
                delta = x-mean
                variance = .9*(variance + .1*delta**2)
                mean = mean + .1*delta
            events = []
            state_cls = sn.RunningStatistics if condition == 'streaming_running' else sn.EMAStatistics
            update = state_cls.update
            def tracked_update(state, x):
                events.append('update')
                return update(state, x)
            def inspect(index, count, mean, std, z, pred):
                events.append('predict')
                self.assertEqual(count, len(initial)+index)
                for actual, ref in zip([mean, std, z, pred], expected[index]):
                    np.testing.assert_allclose(actual, ref)
            with patch.object(state_cls, 'update', tracked_update):
                _, counts = sn.predict_stream(pipe, initial, stream, condition, observer=inspect)
            self.assertEqual(events, ['predict', 'update']*len(stream))
            np.testing.assert_array_equal(counts, [2, 3, 4, 5])
        # Detect a current-sample-in-statistics error with a non-degenerate prediction fixture.
        initial, stream = np.array([[20.]]), np.array([[20.1]])
        correct, _ = sn.predict_stream(pipe, initial, stream, 'streaming_running')
        wrong_state = sn.RunningStatistics(initial)
        wrong_state.update(stream[0])
        wrong_z = (stream-wrong_state.mean)/wrong_state.std()
        wrong = pipe['clf'].predict(pipe['select'].transform(wrong_z))
        self.assertFalse(np.array_equal(correct, wrong))

    def test_future_perturbation_every_suffix_and_last_window(self):
        pipe = self.small_model()
        initial = np.array([[20.], [22.]])
        stream = np.array([[20.5], [21.5], [22.], [20.], [23.]])
        for condition in ['streaming_running', 'streaming_ema']:
            snapshots = []
            original, _ = sn.predict_stream(pipe, initial, stream, condition,
                                            observer=lambda *args: snapshots.append(args))
            # Includes changing only the last window and all t+1...end suffixes.
            for first_changed in range(1, len(stream)):
                changed = stream.copy()
                changed[first_changed:] = 1e8 + np.arange(len(stream)-first_changed).reshape(-1, 1)
                after_snapshots = []
                after, _ = sn.predict_stream(pipe, initial, changed, condition,
                                             observer=lambda *args: after_snapshots.append(args))
                np.testing.assert_array_equal(after[:first_changed], original[:first_changed])
                for before, later in zip(snapshots[:first_changed], after_snapshots[:first_changed]):
                    for a, b in zip(before, later):
                        np.testing.assert_array_equal(a, b)
            # At t itself the input may change, but its pre-prediction statistics cannot.
            changed = stream.copy()
            changed[2] = 1e8
            snapshots_changed = []
            sn.predict_stream(pipe, initial, changed, condition, observer=lambda *args: snapshots_changed.append(args))
            for a, b in zip(snapshots[2][1:4], snapshots_changed[2][1:4]):
                np.testing.assert_array_equal(a, b)

    def test_end_to_end_fit_state_scaler_bypass_and_baseline_equivalence(self):
        original = sn.predict_stream
        build = sn.base.build_pipeline
        fitted, state_by_pipe = [], {}
        with ExitStack() as stack:
            def tracked_build(n_features, params):
                pipe = build(n_features, params)
                spy = stack.enter_context(patch.object(pipe, 'fit', wraps=pipe.fit))
                fitted.append((pipe, spy))
                return pipe
            def guarded(pipe, initial, stream, condition, alpha):
                steps = [pipe[k] for k in ['scaler', 'select', 'clf']]
                state = [pickle.dumps(s.__dict__) for s in steps]
                if condition == 'zero_cal':
                    state_by_pipe[id(pipe)] = (steps, state)
                else:
                    old_steps, old_state = state_by_pipe[id(pipe)]
                    self.assertEqual(state, old_state)
                    for a, b in zip(steps, old_steps):
                        self.assertIs(a, b)
                with ExitStack() as guard:
                    for step in [pipe] + steps:
                        guard.enter_context(patch.object(step, 'fit', side_effect=AssertionError('target refit')))
                    if condition != 'zero_cal':
                        guard.enter_context(patch.object(pipe['scaler'], 'transform', side_effect=AssertionError('double scaling')))
                        guard.enter_context(patch.object(pipe, 'predict', side_effect=AssertionError('calibrated pipeline')))
                    result = original(pipe, initial, stream, condition, alpha)
                self.assertEqual(state, [pickle.dumps(s.__dict__) for s in steps])
                return result
            stack.enter_context(patch.object(sn.base, 'build_pipeline', side_effect=tracked_build))
            stack.enter_context(patch.object(sn, 'predict_stream', side_effect=guarded))
            results = sn.evaluate(self.df, self.params, 'pilot')
            expected = [(cols, train) for cols in sn.base.feature_sets().values()
                        for _, train, _, _ in sn.cs.checked_splits(self.df)]
            self.assertEqual(len(fitted), 9)
            for (pipe, spy), (cols, train) in zip(fitted, expected):
                spy.assert_called_once()
                X = self.df.iloc[train][cols].to_numpy()
                np.testing.assert_array_equal(spy.call_args.args[0], X)
                np.testing.assert_array_equal(spy.call_args.args[1], self.df.iloc[train]['class'])
                np.testing.assert_allclose(pipe['scaler'].mean_, X.mean(axis=0))
        reference = sn.cs.evaluate(self.df, self.params, 'pilot')
        key = ['dataset'] + sn.base.KEY
        for new, old in [('zero_cal', 'zero_cal'), ('fixed_session1', 'session1_cal')]:
            actual = results[2][results[2].condition == new].sort_values(key)
            ref = reference[2][reference[2].condition == old].sort_values(key)
            np.testing.assert_array_equal(actual.y_pred, ref.y_pred)
            a = results[0][results[0].condition == new].sort_values(['dataset', 'subject_num'])
            b = reference[0][reference[0].condition == old].sort_values(['dataset', 'subject_num'])
            np.testing.assert_array_equal(a[['accuracy', 'macro_f1', 'weighted_f1']], b[['accuracy', 'macro_f1', 'weighted_f1']])
        self.assertEqual(len(results[0]), 36)
        self.assertEqual({k: len(v) for k, v in sn.base.feature_sets().items()}, {'Spectral': 620, 'Connectivity': 350, 'Fusion': 970})
        self.assertTrue(np.isfinite(results[2].y_pred).all())
        self.assertEqual(len(results[2]), (self.df.session_index == 3).sum()*12)
        with tempfile.TemporaryDirectory(dir=sn.STAGE / 'output') as tmp:
            sn.save_results(results, Path(tmp)/'output', Path(tmp)/'figures')
            self.assertEqual({p.name for p in (Path(tmp)/'figures').glob('*.png')},
                             {'streaming_accuracy.png', 'streaming_macro_f1.png', 'cumulative_accuracy_vs_time.png'})
            self.assertEqual(len(list((Path(tmp)/'output').glob('*.csv'))), 5)

    def test_splits_chronology_matched_rows_counts_gains_and_summary(self):
        per, summary, pred, manifest = sn.evaluate(self.df, self.params, 'pilot', ['Spectral'])
        plans = sn.chronological_plans(self.df, sn.cs.checked_splits(self.df))
        self.assertNotIn('class', manifest)
        self.assertNotIn('class_label', manifest)
        for subject, train, initial, test in plans:
            self.assertTrue((self.df.iloc[train].subject_num != subject).all())
            self.assertTrue(self.df.iloc[train].session_index.isin([1, 2]).all())
            self.assertTrue((self.df.iloc[initial].subject_num == subject).all())
            self.assertTrue((self.df.iloc[initial].session_index == 1).all())
            self.assertTrue((self.df.iloc[test].session_index == 3).all())
            expected = self.df[(self.df.subject_num == subject) & (self.df.session_index == 3)].sort_values(['trial', 'window_id'])
            np.testing.assert_array_equal(test, expected.index)
            excluded = manifest[(manifest.held_out_subject == subject) & (manifest.subject_num == subject) & (manifest.session_index == 2)]
            self.assertTrue((excluded.role == 'excluded_heldout_session2').all())
            scores = per[per.subject_num == subject].set_index('condition')
            for condition in sn.CONDITIONS:
                block = pred[(pred.subject_num == subject) & (pred.condition == condition)]
                np.testing.assert_array_equal(block[sn.base.KEY], expected[sn.base.KEY])
                np.testing.assert_array_equal(block.chronological_index, np.arange(1, len(test)+1))
                history = np.arange(len(initial), len(initial)+len(test)) if condition.startswith('streaming_') else np.full(len(test), 0 if condition == 'zero_cal' else len(initial))
                np.testing.assert_array_equal(block.history_count_before_prediction, history)
                np.testing.assert_allclose(block.cumulative_accuracy, np.cumsum(block.y_true == block.y_pred)/np.arange(1, len(test)+1))
                self.assertAlmostEqual(scores.loc[condition, 'gain_vs_zero_pp'], 100*(scores.loc[condition, 'accuracy']-scores.loc['zero_cal', 'accuracy']))
                self.assertAlmostEqual(scores.loc[condition, 'gain_vs_fixed_pp'], 100*(scores.loc[condition, 'accuracy']-scores.loc['fixed_session1', 'accuracy']))
        for _, row in summary.iterrows():
            block = per[per.condition == row.condition]
            for metric in sn.METRICS:
                self.assertAlmostEqual(row[f'mean_{metric}'], block[metric].mean())
                self.assertAlmostEqual(row[f'std_{metric}'], block[metric].std(ddof=1))
        curve = sn.cumulative_curve(pred)
        self.assertEqual(set(curve[curve.chronological_index == 1].n_subjects), {3})
        self.assertEqual(set(curve[curve.chronological_index == 12].n_subjects), {1})
        shuffled = sn.evaluate(self.df.sample(frac=1, random_state=9), self.params, 'pilot', ['Spectral'])
        key = ['dataset', 'condition'] + sn.base.KEY
        np.testing.assert_array_equal(pred.sort_values(key).y_pred, shuffled[2].sort_values(key).y_pred)

    def test_session1_and_test_labels_do_not_change_predictions(self):
        original = sn.evaluate(self.df, self.params, 'pilot', ['Spectral'])[2]
        for session in [1, 3]:
            changed = self.df.copy()
            mask = (changed.subject_num == 1) & (changed.session_index == session)
            changed.loc[mask, 'class'] = (changed.loc[mask, 'class']+1) % 3
            changed.loc[mask, 'class_label'] = 'unused'
            later = sn.evaluate(changed, self.params, 'pilot', ['Spectral'])[2]
            # Subject 1 rows train other folds, so compare its held-out fold only.
            for column in ['y_pred', 'history_count_before_prediction']:
                np.testing.assert_array_equal(original[original.subject_num == 1][column], later[later.subject_num == 1][column])

    def test_end_to_end_future_and_heldout_session2_perturbation(self):
        original = sn.evaluate(self.df, self.params, 'pilot', ['Spectral'])
        test = self.df[(self.df.subject_num == 1) & (self.df.session_index == 3)].sort_values(['trial', 'window_id']).index
        for changed_rows in [test[-1:], test[3:], self.df.index[(self.df.subject_num == 1) & (self.df.session_index == 2)]]:
            changed = self.df.copy()
            changed.loc[changed_rows, sn.base.feature_sets()['Fusion']] = 1e8
            later = sn.evaluate(changed, self.params, 'pilot', ['Spectral'])
            before = original[2][original[2].subject_num == 1]
            after = later[2][later[2].subject_num == 1]
            if set(changed_rows) & set(test):
                first = list(test).index(changed_rows[0])+1
                before = before[before.chronological_index < first]
                after = after[after.chronological_index < first]
            pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))

    def test_rejections_before_fit_and_no_initialization_fallback(self):
        with patch.object(sn.base, 'build_pipeline', side_effect=AssertionError('fit before rejection')):
            for session in [1, 3]:
                missing = self.df[~((self.df.subject_num == 3) & (self.df.session_index == session))]
                with self.assertRaisesRegex(ValueError, 'Session 1 and Session 3'):
                    sn.evaluate(missing, self.params, 'pilot')
            with self.assertRaisesRegex(ValueError, 'validated ingestion'):
                sn.evaluate(self.df, self.params, 'final')
            with self.assertRaisesRegex(ValueError, '15 COMPLETE'):
                sn.evaluate(self.df, self.params, 'final', audit={'complete_subjects': []})
        pipe = self.small_model()
        for condition in sn.CONDITIONS[1:]:
            with self.assertRaisesRegex(ValueError, 'initialization required'):
                sn.predict_stream(pipe, np.empty((0, 1)), np.array([[2.]]), condition)
        for alpha in [0, -1, 1.1, np.nan, np.inf]:
            with self.assertRaisesRegex(ValueError, 'EMA alpha'):
                sn.predict_stream(pipe, None, np.array([[2.]]), 'zero_cal', alpha)
        for cls in [sn.RunningStatistics, sn.EMAStatistics]:
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                cls(np.array([[np.inf]]))

    def test_condition_anchors_and_zero_only(self):
        self.assertEqual(sn.canonical_conditions(['streaming_ema']), ['zero_cal', 'fixed_session1', 'streaming_ema'])
        result = sn.evaluate(self.df, self.params, 'pilot', ['Spectral'], ['zero_cal'])
        self.assertEqual(set(result[0].condition), {'zero_cal'})
        self.assertTrue(result[0].gain_vs_fixed_pp.isna().all())
        self.assertTrue((result[2].history_count_before_prediction == 0).all())

    def test_audit_only_no_model_and_fixed_alpha_provenance(self):
        with tempfile.TemporaryDirectory(dir=sn.STAGE / 'output') as tmp:
            with (patch.object(sn, 'STAGE', Path(tmp)),
                  patch.object(sn, 'evaluate', side_effect=AssertionError('real-data experiment')),
                  patch.object(sn.base, 'build_pipeline', side_effect=AssertionError('real-data fit'))):
                self.assertEqual(sn.main(['--audit-only', '--ema-alpha', '0.2']), 0)
            reports = list(Path(tmp).glob('output/audit/*/run_report.json'))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            self.assertEqual(report['status'], 'audit_only')
            self.assertEqual(report['ema_alpha'], .2)
            self.assertEqual(report['ema_alpha_selection'], 'pre-specified fixed, not tuned')
            self.assertTrue((reports[0].parent/'coverage_report.json').exists())
            self.assertFalse(list(Path(tmp).rglob('*.csv')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
