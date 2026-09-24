# Phase 6.5 sanity results

Validated on 2026-09-24. All **12 synthetic unit/sanity tests passed in 6.199 seconds**
(exit code 0). Fixture: three subjects, three sessions, three classes, 81 windows,
970 features, unequal subject stream lengths and shuffled input rows. End-to-end
evaluation covers all datasets and four conditions: 36 score rows and 324 predictions.

Verified:

- Reused 6.4 splits: other subjects S1-S2 train, held-out S1 initialize, held-out S3
  test, held-out S2 excluded. S3 predictions follow trial/window chronology.
- Exactly one pipeline fit per dataset/fold on the exact training arrays/labels.
  Scaler means match only training rows; feature counts remain 620/350/970.
- Zero/fixed predictions, accuracy, macro F1 and weighted F1 exactly match 6.4
  zero_cal/session1_cal on the same input and splits.
- Mock guards prohibit fitting any pipeline step during target inference and prohibit
  training-scaler transform/pipeline.predict for fixed and streaming targets.
  Fitted scaler/selector/classifier objects and serialized states remain unchanged
  across conditions and inference, including support vectors.
- Welford count/mean/variance/std match NumPy population moments at every history
  prefix within 1e-8 absolute tolerance with a 1e6 feature offset. Stored state stays
  count plus two feature vectors, with no sample history.
- EMA initialization and two sequential updates match hand-calculated mean/variance
  values. Std fallback does not overwrite the stored zero variance with 1.
- Strict alternating prediction/update events; moments and normalized inputs match
  S1 + only preceding S3 windows. First prediction uses S1 only.
- A fixture produces different predictions when the current sample is incorrectly
  included before normalization; implemented running inference follows predict-first.
- Future perturbation test: replacing the last window or each future suffix with
  extreme finite features leaves every earlier prediction, pre-prediction statistic
  and normalized input exactly unchanged for both running and EMA.
- Perturbing current x_t leaves its own pre-prediction statistics unchanged. End-to-end
  perturbation repeats last-window/suffix checks through the evaluation runner.
- Changing held-out S1 labels or S3 test labels cannot change its predictions or
  history counts. Changing held-out S2 features cannot change its fold's output.
- All conditions score exact same S3 window keys; finite/complete predictions;
  before-prediction counts are 0, N or N+t-1 as specified. Both gains are matched-row.
- Summary metrics use equal subject weights and sample SD. Per-subject cumulative
  accuracy is correct, and contributor counts at the curve tail reflect stream length.
- Missing initialization/S3/final coverage and invalid alpha are rejected; no fallback.
  Zero-only runs compute no target statistics; requested streaming includes both anchors.
- Four main CSVs, the cumulative-curve CSV, and all three figures are generated in
  temporary 6.5 directories and automatically removed after tests.
- Audit-only cannot call evaluate/build a classifier. Explicit CLI alpha is recorded
  as fixed, not tuned (the audit provenance test uses 0.2 without classification).

Commands completed successfully:

```text
python -B phase_6_wearable_evaluation/6.5_streaming_normalization/code/test_streaming_normalization.py
python -B phase_6_wearable_evaluation/6.5_streaming_normalization/code/run_streaming_normalization.py --audit-only
```

The standalone real-data audit uses the primary default **fixed alpha=0.1**. It finds
only S01 / Session 1 (20131027) / trial 01, with three complete retained windows.
S01 S2/S3 are absent; zero complete subjects. Pilot needs at least two subjects and
S1/S3 per subject; final needs 15 complete subjects and all 675 trials (674 missing).
Both remain ineligible. Audit success indicates inspection completed, not eligibility.

Reports: [coverage](output/audit/20260924T024222_152166Z/coverage_report.json),
[run](output/audit/20260924T024222_152166Z/run_report.json).

No real-data classification, full experiment, feature extraction or alpha optimization
was run. No files outside Phase 6.5 were edited. The schema supports paper tables,
matched subject-level gains, causal history auditing and cumulative curves; final
empirical paper results are not yet available.

Do not optimize EMA alpha on the held-out test subject. The primary result must use a pre-specified alpha and clearly label it as fixed, not tuned.
