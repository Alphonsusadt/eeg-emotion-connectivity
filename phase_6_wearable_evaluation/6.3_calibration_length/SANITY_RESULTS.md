# Phase 6.3 sanity results

Seven synthetic unit tests passed (5.7 seconds). Synthetic data: three subjects,
two sessions, three classes, 54 windows, 970 features. The end-to-end test exercised
all three datasets and seven conditions (63 subject/dataset/condition combinations).

Verified:

- Exact chronological selection on shuffled rows and exclusion from limited/session1 scoring.
- Metadata-only selection API; changing held-out calibration labels does not change
  predictions for that held-out subject or its limited-condition score.
- Calibration-only population mean/std, deterministic zero-variance handling, and no
  statistics for zero calibration.
- No subject/calibration row overlap with classifier training; train-only scaler and ANOVA.
- Exact 0-minute prediction, accuracy, macro-F1 and weighted-F1 equivalence with 6.2.
- Full-subject oracle flags; gain/recovery calculation and NaN for non-positive/tiny denominators.
- Rejection of empty evaluation sets before fitting; final coverage gate and audit-only behavior.
- Synthetic CSV and plot generation. Temporary synthetic files were removed automatically.

Real-data audit only: S01 / session 20131027 / trial 01, three windows, zero complete
subjects. Conditions 0, 1, 2 and full have non-empty evaluation rows within this trial;
5, 10 and session1 are unavailable. LOSO still requires at least two subjects.
Final requires all 675 passed trials (15 subjects × 3 sessions × 15 trials), so 674
additional compatible trials are currently needed.

Reports: [coverage](output/audit/20260923T175702_111805Z/coverage_report.json),
[run](output/audit/20260923T175702_111805Z/run_report.json).

No real-data classification, full experiment, tuning, or EEG extraction was run.
No files outside Phase 6.3 were edited.
