# Phase 6.4 sanity results

Validated on 2026-09-24. Ten synthetic unit/sanity tests passed in 7.959 seconds
(exit code 0). Synthetic fixture: three subjects, three sessions, three classes,
81 windows with unequal subject window counts, and all 970 features. End-to-end
evaluation covers all three datasets and both conditions (18 score rows, 162
window predictions); separate coverage-only metadata covers all 675 expected trials.

Verified:

- Exact other-subject S1-S2 training, held-out S1 calibration, held-out S3 testing;
  subject and row disjointness. Held-out S2 and other-subject S3 are excluded.
- Label-free per-window split manifest records every role and exclusion per fold/condition.
- Exactly one fit per dataset/fold, using the exact ordered training features/labels;
  scaler mean matches only the restricted training rows.
- Zero-cal predictions equal independently fitted ordinary train-pipeline inference;
  no target mean/std is computed for zero calibration.
- Calibration reuses corrected Phase 6.3 inference; mock guards forbid training-scaler
  transform, pipeline.predict, or any fit during calibrated target inference.
- Same fitted objects across conditions; serialized scaler/selector/classifier state
  (including support vectors) remains identical before/after inference.
- Rotating held-out S1 labels does not change its normalization statistics, predictions
  or scores. Test labels are unchanged in this fixture.
- Replacing held-out S2 features with 1e9 and rotating its labels leaves that held-out
  subject's predictions and scores exactly unchanged for all datasets/conditions.
  Other folds are not compared: these rows legitimately train other held-out subjects.
- Feature counts 620/350/970, consistent ordering despite reversed DataFrame columns,
  finite transformed values/predictions, complete and unique S3 predictions.
- Exact same S3 window keys for both conditions; gain_pp equals the paired accuracy
  difference in percentage points. Summary means and sample SD use equal subject weights.
- Missing S1/S3, insufficient subjects, duplicate keys, missing training classes,
  non-finite features and incomplete final coverage are rejected. No silent skipping.
- Complete synthetic session coverage is accepted; a missing final S3 trial is rejected.
- Single zero-only condition and automatic zero anchor for calibrated-only requests.
- Four synthetic CSV outputs and both requested plots are generated in temporary
  directories inside 6.4, then removed automatically.
- Audit-only never calls evaluate or builds a classifier and writes only JSON reports.

Commands run successfully:

```text
python -B phase_6_wearable_evaluation/6.4_cross_session/code/test_cross_session.py
python -B phase_6_wearable_evaluation/6.4_cross_session/code/run_cross_session.py --audit-only
```

Real-data audit: only S01 / S1 (20131027) / trial 01 is available, with three full
windows. S01 S2 and S3 are absent. Zero complete subjects; pilot is ineligible
(needs at least two subjects and S1/S3 for each); final is ineligible (needs 15
complete subjects / 675 passed trials). There are currently 674 missing trials.

Reports: [coverage](output/audit/20260924T022400_070630Z/coverage_report.json),
[run](output/audit/20260924T022400_070630Z/run_report.json).

No real-data classification, full experiment, parameter tuning or feature extraction
was run. No files outside Phase 6.4 were edited. The output schema is ready for
paper tables and subject-paired comparisons, but no final empirical result is available.
