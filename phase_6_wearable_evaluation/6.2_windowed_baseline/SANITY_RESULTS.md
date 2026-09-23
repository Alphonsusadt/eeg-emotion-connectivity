# Implementation sanity results

- Eight synthetic unit/sanity tests passed (12.5 seconds).
- Synthetic LOSO: 3 subjects, 3 classes, 18 windows; all three feature sets exercised.
- Verified train-only scaler means and ANOVA scores against training rows.
- Verified single-subject test folds, no subject overlap, complete out-of-fold predictions,
  expected feature counts, finite values, and metadata exclusion.
- Verified rerun deduplication, non-passed exclusion, window-boundary rejection,
  complete-subject detection, and rejection of 15 present but incomplete subjects.
- Verified CSV/figure output generation on synthetic data in temporary directories.
- Real input audit only: S01, session 20131027, trial 01; 3 valid windows.
- Complete subjects: 0/15. S01 has 1/45 passed trials; S02–S15 have 0/45.
- Final still needs 674 additional passed trials (assuming the existing trial stays compatible).
- Pilot LOSO also cannot run yet: only one subject is available.
- No real-data classifier fitting, new hyperparameter tuning, or EEG extraction performed.

Audit details: [coverage_report.json](output/audit/20260923T172837_010572Z/coverage_report.json).
Unit-test temporary predictions/figures were removed automatically; they are not research results.
