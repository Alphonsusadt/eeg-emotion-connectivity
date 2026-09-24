# Phase 6.3 sanity results

Ten synthetic unit tests passed (7.235 seconds) on 2026-09-24 after the scaling fix. Synthetic data: three subjects,
two sessions, three classes, 54 windows, 970 features. The end-to-end test exercised
all three datasets and seven conditions (63 subject/dataset/condition combinations).

Verified:

- Exact chronological selection on shuffled rows and exclusion from limited/session1 scoring.
- Metadata-only selection API; changing held-out calibration labels does not change
  predictions for that held-out subject or its limited-condition score.
- Calibration-only population mean/std, deterministic zero-variance handling, and no
  statistics for zero calibration.
- No subject/calibration row overlap with classifier training; train-only scaler and ANOVA.
- Calibrated prediction cannot call the training scaler or pipeline.predict (mock guards
  raise immediately); all seven conditions and all three datasets are exercised.
- The original fitted selector and classifier receive the expected target z-scores and
  selected arrays. No target-time fit is allowed. Serialized estimator state is identical
  before/after prediction and across conditions, including scaler parameters, selector
  parameters/scores and classifier parameters/support vectors.
- Changing all held-out labels leaves chronology selection, normalization statistics and
  transformed values identical for every condition. Transformed/selected values are finite.
- One-window calibration replaces every zero std with 1 and produces exactly x - mu.
- A shifted-mean synthetic SVM fixture explicitly gives different predictions for the
  wrong target_z -> training scaler -> selector -> SVM path and the correct
  target_z -> fitted selector -> fitted SVM path; the implementation matches the latter.
- Exact 0-minute prediction, accuracy, macro-F1 and weighted-F1 equivalence with 6.2.
- Full-subject oracle flags; gain/recovery calculation and NaN for non-positive/tiny denominators.
- Rejection of empty evaluation sets before fitting; final coverage gate and audit-only behavior.
- Synthetic CSV and plot generation. Temporary synthetic files were removed automatically.

Real-data audit only: S01 / session 20131027 / trial 01, three windows, zero complete
subjects. Conditions 0, 1, 2 and full have non-empty evaluation rows within this trial;
5, 10 and session1 are unavailable. LOSO still requires at least two subjects.
Final requires all 675 passed trials (15 subjects × 3 sessions × 15 trials), so 674
additional compatible trials are currently needed.

Reports: [coverage](output/audit/20260924T015515_945891Z/coverage_report.json),
[run](output/audit/20260924T015515_945891Z/run_report.json).

No real-data classification, full experiment, tuning, or EEG extraction was run.
No files outside Phase 6.3 were edited.

Commands completed successfully (exit code 0):

```text
python -B phase_6_wearable_evaluation/6.3_calibration_length/code/test_calibration_length.py
python -B phase_6_wearable_evaluation/6.3_calibration_length/code/run_calibration_length.py --audit-only
```

The final-coverage rejection printed during unit tests is expected and asserted.
Output schemas and metric definitions (including matched-row metrics) are unchanged.
The existing run-report calibration_space description now states that calibrated
prediction bypasses the training scaler. Historical 6.3 outputs were not edited.
