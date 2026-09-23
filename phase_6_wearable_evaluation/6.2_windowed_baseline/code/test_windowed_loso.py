"""Synthetic regression tests; no real EEG classification or extraction."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

import run_windowed_loso as runner


def synthetic_data():
    rng = np.random.default_rng(42)
    rows = []
    for subject in [1, 2, 3]:
        for trial in [1, 2, 3]:
            raw = runner.cfg.TRIAL_LABELS[trial - 1]
            for window in [1, 2]:
                row = dict(zip(runner.feature_sets()['Fusion'], rng.normal(subject * 10, 1, 970)))
                row.update(subject_id=f'S{subject:02d}', subject_num=subject,
                           session=int(runner.cfg.SUBJECT_SESSIONS[subject][0]), session_index=1,
                           trial=trial, window_id=window, start_sample=(window - 1) * 6000,
                           stop_sample=window * 6000, start_seconds=(window - 1) * 60,
                           stop_seconds=window * 60, **{'class': runner.cfg.LABEL_MAP[raw],
                           'class_label': runner.cfg.EMOTION_MAP[raw]})
                rows.append(row)
    return pd.DataFrame(rows)


def write_run(root, frame, run='run_001', status='passed'):
    row = frame.iloc[0]
    folder = root / f'subject_{int(row.subject_num):02d}' / f'session_{int(row.session)}' / f'trial_{int(row.trial):02d}' / run
    folder.mkdir(parents=True)
    frame.to_csv(folder / 'windowed_features.csv', index=False)
    info = {'status': status, 'fs': 100, 'window_seconds': 60, 'overlap_samples': 0,
            'tail_policy': 'drop', 'input_shape': [62, 12001], 'channels': runner.cfg.CHANNEL_NAMES,
            'n_windows': 2, 'completed_windows': 2, 'n_features': 970, 'discarded_samples': 1,
            'pipeline_sha256': {'fixture': 'synthetic'}, 'gc_max_lag': 10, 'gc_threshold': 'fdr',
            'gc_alpha': .05, 'pdc_max_order': 15, 'pdc_min_order': 3,
            'bands': runner.cfg.PDC_FREQUENCY_BANDS, 'matrix_orientation': 'fixture', 'normalization': 'none'}
    report = folder / 'sanity_report.json'
    report.write_text(json.dumps(info), encoding='utf-8')
    return report


class WindowedLosoTests(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_data()
        self.params = {name: {'C': 1, 'gamma': 'scale'} for name in runner.PARAM_SOURCE}

    def test_schema_finite_and_metadata(self):
        sets = runner.validate_features(self.df)
        self.assertEqual([len(v) for v in sets.values()], [620, 350, 970])
        self.assertFalse(set(sets['Fusion']) & set(runner.META))
        for bad in [self.df.assign(unexpected_metadata=1), self.df.rename(columns={sets['Spectral'][0]: 'spec_label'})]:
            with self.assertRaises(ValueError):
                runner.validate_features(bad)
        for value in [np.nan, np.inf]:
            bad = self.df.copy()
            bad.loc[0, sets['Spectral'][0]] = value
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                runner.validate_features(bad)

    def test_fold_isolation_and_train_fitted_statistics(self):
        seen = []
        def inspect(name, pipe, train, test):
            cols = runner.feature_sets()[name]
            X_train = self.df.iloc[train][cols].to_numpy()
            np.testing.assert_allclose(pipe['scaler'].mean_, X_train.mean(axis=0))
            self.assertFalse(np.allclose(pipe['scaler'].mean_, self.df[cols].mean()))
            expected_scores, _ = f_classif(pipe['scaler'].transform(X_train), self.df.iloc[train]['class'])
            np.testing.assert_allclose(pipe['select'].scores_, expected_scores)
            self.assertFalse(set(self.df.iloc[train].subject_num) & set(self.df.iloc[test].subject_num))
            self.assertEqual(self.df.iloc[test].subject_num.nunique(), 1)
            seen.append(name)
        results = runner.evaluate(self.df, self.params, 'pilot', observer=inspect)
        self.assertEqual(len(seen), 9)
        self.assertEqual(len(results[0]), 9)
        self.assertEqual(len(results[2]), len(self.df) * 3)
        self.assertTrue((results[0].n_test_windows == 6).all())
        with tempfile.TemporaryDirectory(dir=runner.STAGE / 'output') as tmp:
            root = Path(tmp)
            runner.save_results(results, root / 'output', root / 'figures')
            self.assertEqual(len(list((root / 'output').glob('*.csv'))), 3)
            self.assertEqual(len(list((root / 'figures').glob('*.png'))), 2)

    def test_test_values_do_not_change_training_fit(self):
        train, test = runner.checked_splits(self.df)[0]
        cols = runner.feature_sets()['Spectral']
        X = self.df[cols].to_numpy()
        a = runner.build_pipeline(len(cols), self.params['Spectral']).fit(X[train], self.df.iloc[train]['class'])
        X[test] += 1e8
        b = runner.build_pipeline(len(cols), self.params['Spectral']).fit(X[train], self.df.iloc[train]['class'])
        np.testing.assert_array_equal(a['scaler'].mean_, b['scaler'].mean_)
        np.testing.assert_array_equal(a['select'].scores_, b['select'].scores_)

    def test_coverage_gates(self):
        audit = {'complete_subjects': []}
        with self.assertRaisesRegex(ValueError, '15 COMPLETE'):
            runner.eligibility(self.df, audit)
        runner.eligibility(self.df, audit, pilot=True)
        with self.assertRaisesRegex(ValueError, 'at least two'):
            runner.eligibility(self.df[self.df.subject_num == 1], audit, pilot=True)
        missing_class = self.df[self.df['class'] != 0]
        with self.assertRaisesRegex(ValueError, 'all three classes'):
            runner.checked_splits(missing_class)

    def test_discovery_duplicate_runs_and_incomplete_coverage(self):
        frame = self.df[(self.df.subject_num == 1) & (self.df.trial == 1)]
        with tempfile.TemporaryDirectory(dir=runner.STAGE / 'output') as tmp:
            root = Path(tmp)
            write_run(root, frame)
            write_run(root, frame, 'run_002')
            write_run(root, frame, 'run_003', 'running')
            merged, audit = runner.collect(root)
            self.assertEqual(len(merged), 2)
            self.assertEqual(len(audit['superseded_passed']), 1)
            self.assertEqual(len(audit['ignored_nonpassed']), 1)
            self.assertEqual(audit['complete_subjects'], [])
            self.assertEqual(audit['coverage'][0]['passed_trials'], 1)
            self.assertEqual(len(audit['coverage'][0]['missing_trials']), 44)
            latest = Path(audit['selected_runs'][0]['report'])
            bad = pd.read_csv(latest.parent / 'windowed_features.csv')
            bad.loc[bad.index[0], 'start_sample'] = 100
            bad.to_csv(latest.parent / 'windowed_features.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'gap/overlap'):
                runner.collect(root)

    def test_all_15_present_but_partial_is_not_final(self):
        with tempfile.TemporaryDirectory(dir=runner.STAGE / 'output') as tmp:
            for subject in runner.cfg.SUBJECT_SESSIONS:
                frame = self.df[(self.df.subject_num == 1) & (self.df.trial == 1)].copy()
                frame.subject_num = subject
                frame.subject_id = f'S{subject:02d}'
                frame.session = int(runner.cfg.SUBJECT_SESSIONS[subject][0])
                write_run(Path(tmp), frame)
            merged, audit = runner.collect(Path(tmp))
            self.assertEqual(len(audit['available_subjects']), 15)
            with self.assertRaisesRegex(ValueError, '15 COMPLETE'):
                runner.eligibility(merged, audit)

    def test_complete_subject_detected(self):
        with tempfile.TemporaryDirectory(dir=runner.STAGE / 'output') as tmp:
            for index, date in enumerate(runner.cfg.SUBJECT_SESSIONS[1], 1):
                for trial in range(1, 16):
                    frame = self.df[(self.df.subject_num == 1) & (self.df.trial == 1)].copy()
                    raw = runner.cfg.TRIAL_LABELS[trial - 1]
                    frame.session, frame.session_index, frame.trial = int(date), index, trial
                    frame['class'], frame['class_label'] = runner.cfg.LABEL_MAP[raw], runner.cfg.EMOTION_MAP[raw]
                    write_run(Path(tmp), frame)
            _, audit = runner.collect(Path(tmp))
            self.assertEqual(audit['complete_subjects'], [1])

    def test_audit_never_fits(self):
        with tempfile.TemporaryDirectory(dir=runner.STAGE / 'output') as tmp:
            with patch.object(runner, 'STAGE', Path(tmp)), patch.object(runner, 'evaluate', side_effect=AssertionError('must not fit')):
                self.assertEqual(runner.main(['--audit-only']), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
