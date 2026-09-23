"""Synthetic tests only. No real EEG model training."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

import run_calibration_length as cal


def synthetic_data():
    rng = np.random.default_rng(71)
    rows = []
    for subject in [1, 2, 3]:
        for session in [1, 2]:
            for trial in [1, 2, 3]:
                for window in [1, 2, 3]:
                    row = dict(zip(cal.base.feature_sets()['Fusion'], rng.normal(subject * 2, 1, 970)))
                    raw = cal.base.cfg.TRIAL_LABELS[trial - 1]
                    row.update(subject_id=f'S{subject:02d}', subject_num=subject,
                               session=int(cal.base.cfg.SUBJECT_SESSIONS[subject][session - 1]),
                               session_index=session, trial=trial, window_id=window,
                               start_sample=(window - 1)*6000, stop_sample=window*6000,
                               start_seconds=(window - 1)*60, stop_seconds=window*60,
                               **{'class': cal.base.cfg.LABEL_MAP[raw], 'class_label': cal.base.cfg.EMOTION_MAP[raw]})
                    rows.append(row)
    return pd.DataFrame(rows).sample(frac=1, random_state=17).reset_index(drop=True)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_data()
        self.params = {name: {'C': 1, 'gamma': 'scale'} for name in cal.base.PARAM_SOURCE}

    def test_exact_chronology_and_exclusion(self):
        held = self.df[self.df.subject_num == 1].reset_index(drop=True)
        ordered = held.sort_values(['session_index', 'trial', 'window_id']).index.to_numpy()
        for condition in ['1', '2', '5', '10']:
            selected, evaluation = cal.select_rows(held[cal.ORDER], condition)
            np.testing.assert_array_equal(selected, ordered[:int(condition)])
            self.assertFalse(set(selected) & set(evaluation))
            self.assertEqual(len(selected) + len(evaluation), len(held))
        selected, evaluation = cal.select_rows(held[cal.ORDER], 'session1')
        self.assertTrue((held.iloc[selected].session_index == 1).all())
        self.assertTrue((held.iloc[evaluation].session_index == 2).all())
        with self.assertRaisesRegex(ValueError, 'no labels'):
            cal.select_rows(held[cal.ORDER + ['class']], '1')

    def test_statistics_and_zero_variance(self):
        X = np.array([[1., 8.], [3., 8.], [5., 9.]])
        result, mean, std = cal.transform_target(X, np.array([0, 1]), np.array([2]))
        np.testing.assert_array_equal(mean, [2, 8])
        np.testing.assert_array_equal(std, [1, 1])
        np.testing.assert_array_equal(result, [[3, 1]])
        one, _, std = cal.transform_target(X, np.array([0]), np.array([2]))
        np.testing.assert_array_equal(std, [1, 1])
        np.testing.assert_array_equal(one, [[4, 1]])
        zero, mean, std = cal.transform_target(X, np.array([], dtype=int), np.arange(3))
        self.assertIsNone(mean)
        self.assertIsNone(std)
        np.testing.assert_array_equal(zero, X)
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            cal.transform_target(np.array([[np.inf]]), np.array([0]), np.array([0]))

    def test_recovery(self):
        metrics = cal.gain_metrics(.7, .5, .9)
        self.assertAlmostEqual(metrics['gain_pp'], 20)
        self.assertAlmostEqual(metrics['gap_to_full_pp'], 20)
        self.assertAlmostEqual(metrics['recovery_pct'], 50)
        for full in [.4, .5, .5 + 1e-12]:
            self.assertTrue(np.isnan(cal.gain_metrics(.7, .5, full)['recovery_pct']))

    def test_empty_evaluation_rejected_before_fitting(self):
        tiny = self.df.groupby('subject_num', sort=False).head(2)
        with patch.object(cal.base, 'build_pipeline', side_effect=AssertionError('must not fit')):
            with self.assertRaises(ValueError):
                cal.evaluate(tiny, self.params, 'pilot')
        held = self.df[self.df.subject_num == 1]
        held = held[held.session_index == 1]
        with self.assertRaisesRegex(ValueError, 'Empty evaluation'):
            cal.select_rows(held[cal.ORDER], 'session1')
        with self.assertRaisesRegex(ValueError, 'one held-out'):
            cal.select_rows(self.df[cal.ORDER], '1')

    def test_end_to_end_zero_equivalence_and_train_only_fit(self):
        seen = []
        def inspect(name, subject, condition, pipe, train, test, chosen, ev, mu, std):
            columns = cal.base.feature_sets()[name]
            X = self.df[columns].to_numpy()
            self.assertFalse(set(train) & set(test[chosen]))
            self.assertFalse(set(self.df.iloc[train].subject_num) & set(self.df.iloc[test].subject_num))
            np.testing.assert_allclose(pipe['scaler'].mean_, X[train].mean(axis=0))
            scores, _ = f_classif(pipe['scaler'].transform(X[train]), self.df.iloc[train]['class'])
            np.testing.assert_allclose(pipe['select'].scores_, scores)
            if condition not in ['0', 'full']:
                self.assertFalse(set(chosen) & set(ev))
            if condition == '0':
                self.assertIsNone(mu)
            else:
                np.testing.assert_allclose(mu, X[test[chosen]].mean(axis=0))
            seen.append((name, subject, condition))
        results = cal.evaluate(self.df, self.params, 'pilot', observer=inspect)
        per, summary, predictions, curve, selections = results
        old = cal.base.evaluate(self.df, self.params, 'pilot')
        zero = per[per.condition == '0'].sort_values(['dataset', 'subject_num'])
        reference = old[0].sort_values(['dataset', 'subject_num'])
        np.testing.assert_array_equal(zero[['accuracy', 'macro_f1', 'weighted_f1']], reference[['accuracy', 'macro_f1', 'weighted_f1']])
        key = ['dataset'] + cal.base.KEY
        actual_pred = predictions[predictions.condition == '0'].sort_values(key).y_pred.to_numpy()
        np.testing.assert_array_equal(actual_pred, old[2].sort_values(key).y_pred.to_numpy())
        self.assertEqual(len(seen), 63)
        self.assertTrue(per[per.condition == 'full'].oracle_reference.all())
        self.assertFalse(per[per.condition != 'full'].oracle_reference.any())
        self.assertEqual(summary.condition.tolist()[:7], cal.CONDITIONS)
        self.assertTrue((summary.n_subjects == 3).all())
        with tempfile.TemporaryDirectory(dir=cal.STAGE / 'output') as tmp:
            folder = Path(tmp)
            cal.save_results(results, folder / 'output', folder / 'figures')
            self.assertEqual(len(list((folder / 'output').glob('*.csv'))), 5)
            self.assertEqual(len(list((folder / 'figures').glob('*.png'))), 4)

    def test_calibration_labels_cannot_change_predictions(self):
        original = cal.evaluate(self.df, self.params, 'pilot', ['Spectral'], ['2'])
        modified = self.df.copy()
        indices = modified.index[modified.subject_num == 1]
        chosen, _ = cal.select_rows(modified.loc[indices, cal.ORDER], '2')
        modified.loc[indices[chosen], 'class'] = (modified.loc[indices[chosen], 'class'] + 1) % 3
        modified.loc[indices[chosen], 'class_label'] = 'unavailable'
        # Subject 1 is held-out in the fold being compared; its labels may be used
        # as training labels in other LOSO folds, so compare only this fold.
        changed = cal.evaluate(modified, self.params, 'pilot', ['Spectral'], ['2'])
        keys = ['condition'] + cal.base.KEY
        before = original[2][original[2].subject_num == 1].sort_values(keys)
        after = changed[2][changed[2].subject_num == 1].sort_values(keys)
        np.testing.assert_array_equal(before.y_pred, after.y_pred)
        before_score = original[0][(original[0].subject_num == 1) & (original[0].condition == '2')].accuracy.iloc[0]
        after_score = changed[0][(changed[0].subject_num == 1) & (changed[0].condition == '2')].accuracy.iloc[0]
        self.assertEqual(before_score, after_score)

    def test_audit_only_never_fits_and_final_gate(self):
        with tempfile.TemporaryDirectory(dir=cal.STAGE / 'output') as tmp:
            with patch.object(cal, 'STAGE', Path(tmp)), patch.object(cal, 'evaluate', side_effect=AssertionError('no real fit')):
                self.assertEqual(cal.main(['--audit-only']), 0)
                self.assertEqual(cal.main(['--require-all-subjects']), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
