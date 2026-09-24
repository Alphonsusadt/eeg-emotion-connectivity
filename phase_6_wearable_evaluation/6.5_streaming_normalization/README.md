# Phase 6.5: Causal streaming normalization / online unsupervised adaptation

Offline replay of validated SEED 60-second feature windows, using **prequential
evaluation** and **Session-1 initialization**. This is online normalization
adaptation, not online model training: the scaler, selector and SVM fitted on
training subjects are frozen. Only target normalization statistics adapt.

All new code/results live here. Phases 4, 5, 6.1, 6.2, 6.3 and 6.4 remain unchanged.

## Primary split and chronology

| Rows in a held-out subject fold | Role |
|---|---|
| Other subjects, S1-S2 | Raw model training |
| Other subjects, S3 | Excluded |
| Held-out S1 | Unlabeled initialization for fixed/running/EMA; unused for zero_cal |
| Held-out S2 | Excluded from training, initialization, stream and scoring |
| Held-out S3 | Entire test stream, sorted by `trial -> window_id` |

Session indices are defined by `config.SUBJECT_SESSIONS`, with different recording
dates for different subjects. No shuffling of the S3 stream is allowed. State continues
across trials within S3, without resetting at trial boundaries; it resets for each
subject/dataset/condition. Initialization always precedes S3 and never uses S2.
All conditions score the exact same S3 window keys, with no warm-up windows dropped.
The chronological index starts at 1 for the first retained S3 window of each subject.

The runner imports 6.4 without executing its CLI and reuses its `checked_splits`,
`eligibility`, `session_coverage`, and split-manifest assignment. Through 6.4 it reuses
6.2 ingestion, exact schema/feature definitions, fixed parameters and pipeline, and
6.3's corrected `predict_target` for the zero/fixed references. New code implements
the streaming state, chronological inference, streaming metrics and outputs.

## Four conditions and transformation order

Training remains `raw train -> train StandardScaler -> SelectKBest -> SVM`.
One model per dataset/fold is fitted only on other-subject S1-S2 rows, then reused.

| Condition | Statistics used for prediction | Target prediction path |
|---|---|---|
| `zero_cal` | No target statistics | raw S3 -> train StandardScaler -> fitted selector -> fitted SVM |
| `fixed_session1` | Fixed batch mean/population std of all held-out S1 windows | raw S3 -> S1 z-score -> fitted selector -> fitted SVM |
| `streaming_running` | S1 plus only preceding S3 windows, equally weighted | raw S3_t -> causal target z-score -> fitted selector -> fitted SVM |
| `streaming_ema` | S1 batch initialization followed by fixed-alpha EMA updates on preceding S3 windows | raw S3_t -> causal target z-score -> fitted selector -> fitted SVM |

Fixed and streaming targets **never pass through the train StandardScaler** after
target normalization. The feature count/order and fitted selector/classifier are
unchanged. Zero/fixed predictions and metrics are checked against 6.4 zero_cal and
session1_cal on identical splits. The coordinate-space assumption follows 6.3:
target statistics substitute for the training scaler at target inference. This does
not guarantee distribution matching, session stationarity or improved performance.

Spectral = 620, Connectivity = 350, Fusion = 970. Historical SVM_Tuned mapping is
unchanged: Spectral -> Spectral, Connectivity -> GC+PDC_ROI,
Fusion -> GC+PDC_ROI+Spectral. The model is the 6.2 RBF SVC with fixed parameters,
`probability=False`, and `SelectKBest(f_classif, min(1200,n_features))`. No tuning,
pseudo-label training, classifier updates or target-label feature selection occurs.

## Strict causal update rule

At S3 window t (1-based), only S1 and S3 windows 1 through t-1 may affect its statistics:

```text
stats_(t-1)
  -> normalize raw x_t
  -> fitted selector
  -> fitted SVM predict
  -> record prediction and history count
  -> update statistics using raw x_t
  -> stats_t
```

The current sample is not included in its own normalization statistics. Future windows
are never used to estimate them. Evaluation labels are accessed after label-free stream
inference, only for metrics. The predictor API accepts feature arrays, not labels.
Input files are preloaded for this offline replay; normalization state itself stores
only O(n_features) moments, not the sample history. Preloading is not used to compute
future target statistics.

Ingestion retains 6.2's trial-label integrity checks and finite/schema checks. These
checks validate the dataset, not calibration or adaptation choices. Causal claims
apply to normalization of already extracted features, not to the inherited offline
EEG preprocessing/extraction or latency of acquiring a complete 60-second window.

## Running moments and EMA

All equations below apply feature-wise. Both methods initialize with all available
held-out S1 windows using raw batch population moments:

```text
n = number of S1 windows
mu = mean(S1)
M2 = sum((S1 - mu)^2)
variance = M2 / n                 # ddof=0
```

Running updates use Welford, after prediction:

```text
delta = x_t - mu_old
n_new = n_old + 1
mu_new = mu_old + delta / n_new
M2_new = M2_old + delta * (x_t - mu_new)
variance_new = M2_new / n_new
```

Running state stores only `count`, `mean`, `M2`. Tests compare its moments against
NumPy batch statistics on every prefix, including large feature offsets.

EMA starts from the same S1 mean and population variance, then uses a fixed protocol
alpha (default **0.1**, not a tuned optimum). S1 is batch initialized, not itself
processed with EMA. After each S3 prediction:

```text
delta = x_t - mu_old
mu_new = mu_old + alpha * delta
variance_new = (1-alpha) * (variance_old + alpha * delta^2)
count_new = count_old + 1
```

The old mean is used for delta in both EMA recurrences. This is a central-moment
update for an exponentially weighted distribution; no sample-variance/Bessel or
EMA bias correction is applied. The onboarding distribution's total weight decays
by `(1-alpha)` per S3 update. The count tracks observed windows, not EMA effective
sample size. `--ema-alpha` accepts finite `0 < alpha <= 1`; alpha=1 is a degenerate
last-sample mean with zero variance and the same std fallback below. There is no
automatic alpha sweep or performance-based selection.

For prediction, `std = sqrt(max(variance,0))`, with negative roundoff clamped to zero;
`std < 1e-10` is replaced with 1.0, consistent with 6.3. The fallback is applied only
to the normalization denominator; stored variance/M2 remains the genuine estimate.
One initialization window therefore gives an offset-only first transform, not a
stable variance estimate. Non-finite moments/transforms/predictions are rejected.

## Coverage behavior

Ingestion accepts only validated passed 6.1 trials, deduplicates reruns, validates all
full windows against trial input shapes and rejects mixed extraction provenance.
The 6.4 gate is reused unchanged:

- Final/default: all 15 complete subjects, 45 complete passed trials each (675 total),
  including every expected trial/window in S1 and S3. No incomplete subject is skipped.
- Pilot: at least two subjects, S1 and S3 available for every evaluated subject,
  non-empty other-subject S1-S2 training and all three classes in each training fold.
  All available passed S1 windows initialize adaptation; partial S1 is explicitly pilot.
- Missing S1 rejects running/EMA/fixed; no fallback to zero calibration. Following
  6.4, S1/S3 eligibility also applies to zero-only runs to preserve the subject cohort.
- Audit-only: inspect coverage and eligibility without evaluating/fitting any model.
  Audit success is not a statement that pilot or final evaluation is eligible.

S2 is still required by final completeness because it trains other held-out folds,
but is entirely excluded when its own subject is held out.

## Metrics, schema and figures

Unique run directories: `output/{audit,pilot,final}/TIMESTAMP/`; figures go to
`figures/{pilot,final}/TIMESTAMP/`. An experiment writes:

| File | Contents |
|---|---|
| `streaming_per_subject.csv` | mode/protocol/dataset/model/condition, held-out subject_num/subject_id, train_session_set, initialization_session, test_session, ema_alpha/fixed flag, n_features, n_train_windows, n_initialization_windows, n_test_windows, accuracy, macro_f1, weighted_f1, confusion_matrix, both gains |
| `streaming_summary.csv` | one row per dataset/condition, protocol/session/alpha identifiers, n_subjects, total test windows, mean_*, std_* (sample SD) and n_valid_* for metrics/gains |
| `streaming_predictions.csv` | test-window metadata, dataset/condition/protocol/mode/alpha, chronological_index, y_true, y_pred, history_count_before_prediction, cumulative_accuracy |
| `streaming_split_manifest.csv` | all input-window identities and per-fold/condition roles; no labels/features; chronological_index is 0 for non-test rows |
| `streaming_cumulative_curve.csv` | per-dataset/condition/index mean and sample SD of subject cumulative accuracy, plus n_subjects contributing at each index |
| `coverage_report.json` | inherited source paths/hashes, missing trials, session coverage and pilot/final eligibility |
| `run_report.json` | statuses, causal ordering, formulas, initialization, fixed alpha, transforms, exact feature lists, historical parameters and code/config/source hashes |

`train_session_set` is the string `1,2`. Initialization session is 0 (none) for zero_cal,
1 otherwise; test session is 3. `ema_alpha` is a run-level protocol parameter repeated
in tables for provenance; it affects only streaming_ema. `ema_alpha_fixed=True` means
the run does not tune it. Historical parameters and code hashes identify the reference
implementation. No per-window vectors of 970 means/stds are written to the CSVs.

`history_count_before_prediction`, with N = number of S1 windows and t starting at 1:

| Condition | Count before window t |
|---|---|
| zero_cal | 0 |
| fixed_session1 | N, constant |
| streaming_running / streaming_ema | N + t - 1 |

Each condition starts from its own fresh state. The first streaming prediction uses
only S1; the previous raw S3 window is incorporated only after its prediction.
Manifest roles reuse 6.4 exclusions, with `initialization` replacing `calibration`.

Accuracy/F1 are proportions; F1 uses fixed classes [0,1,2], `zero_division=0`.
Confusion matrices are JSON arrays, actual class on rows and prediction on columns.
Both gains are evaluated on identical S3 rows:

```text
gain_vs_zero_pp  = 100 * (Acc_condition - Acc_zero_cal)
gain_vs_fixed_pp = 100 * (Acc_condition - Acc_fixed_session1)
```

Gains are included for all conditions with available anchors (self-gain is zero).
For a zero-only run, gain_vs_fixed_pp is NaN/blank, with zero valid contributors.
Summary means and sample SD (`ddof=1`) give each subject equal weight, not pooled
window weights; SD is not a confidence interval. Anchors are from this run, not old
aggregate results. Requesting streaming automatically adds zero/fixed anchor conditions;
requesting fixed automatically includes zero.

Figures:

- `streaming_accuracy.png` and `streaming_macro_f1.png`: four-condition comparison
  across requested datasets, subject mean +/- sample SD.
- `cumulative_accuracy_vs_time.png`: a panel per dataset, showing each condition's
  mean subject cumulative accuracy against chronological retained 60-s window index.

For each subject, cumulative accuracy at t is correct predictions in 1..t divided by t.
Curves average equally over subjects with a window at that index. No padding or
last-value carry-forward is used; tail subject composition can change with stream
length. `streaming_cumulative_curve.csv` records contributor counts to make this
explicit. This x-axis is retained windows, **not actual wall-clock performance**;
breaks, trial gaps and discarded tails are not represented. Metrics/plots are computed
after inference and never feed back into adaptation.

## CLI and checks

Safe validation from repository root:

```powershell
python -B phase_6_wearable_evaluation/6.5_streaming_normalization/code/test_streaming_normalization.py
python -B phase_6_wearable_evaluation/6.5_streaming_normalization/code/run_streaming_normalization.py --audit-only
```

Future experiments require an explicit request and eligible data:

```powershell
python -B phase_6_wearable_evaluation/6.5_streaming_normalization/code/run_streaming_normalization.py --pilot --datasets Spectral --conditions zero_cal fixed_session1 streaming_running streaming_ema --ema-alpha 0.1
python -B phase_6_wearable_evaluation/6.5_streaming_normalization/code/run_streaming_normalization.py --require-all-subjects --datasets Spectral Connectivity Fusion --ema-alpha 0.1
```

Defaults: primary protocol, all datasets, all four conditions, fixed alpha=0.1,
final coverage gate. Pilot and require-all-subjects flags are mutually exclusive.
Audit writes only coverage/run JSON reports. Synthetic tests create and remove
temporary CSV/figure output inside 6.5. Dependencies are those of 6.2.

Tests cover exact split/chronology; ordinary/fixed baseline equivalence; no refitting
or second scaler pass; Welford prefix moments; manual EMA recurrence; current-sample
exclusion; prediction-before-update events; last-window and all-future-suffix
perturbations; target label and held-out S2 independence; matched metrics and counts.

## Paper interpretation

Use “causal streaming normalization”, “online unsupervised adaptation”, “prequential
evaluation”, and “Session-1 initialization”. Classifier parameters never change.
This offline normalization experiment does not establish an adaptive clinical system
or real-time wearable validation; it is not an online learning classifier.
Inherited offline preprocessing, repeated SEED stimuli and historical hyperparameter
tuning on the same cohort remain limitations. This is not independent/nested model
selection and synthetic tests are not empirical evidence of adaptation benefit.

The schema supports TRON paper tables, paired subject-level gains, causal-history
audits and reproducible curves. Final empirical results still require complete
coverage and a separately requested real-data experiment.

Do not optimize EMA alpha on the held-out test subject. The primary result must use a pre-specified alpha and clearly label it as fixed, not tuned.
