# Phase 6.4: Cross-session subject-independent evaluation

Offline SEED simulation using validated 60-second, non-overlapping feature windows
from Phase 6.1. The comparison is **unlabeled Session-1 calibration** followed by
**held-out Session-3 evaluation**. All code and outputs are confined to this stage.
Phases 4, 5, 6.1, 6.2 and 6.3 are imported/read only.

## Primary protocol

For each held-out subject, use these roles:

| Rows | zero_cal | session1_cal |
|---|---|---|
| Other subjects, Session 1 and Session 2 | Training | Same training/model |
| Other subjects, Session 3 | Excluded | Excluded |
| Held-out subject, Session 1 | Excluded; no target statistics | All available S1 windows, unlabeled calibration |
| Held-out subject, Session 2 | Excluded | Excluded |
| Held-out subject, Session 3 | All S3 windows scored | Exact same S3 windows scored |

Session numbers are `session_index` 1, 2, 3 in `config.SUBJECT_SESSIONS`, not global
calendar dates shared by subjects. Session-date/index consistency is validated by
6.2 ingestion. Only the `primary` protocol and the two conditions above are implemented;
no secondary protocol, oracle normalization, condition search, or parameter tuning.

Every available subject is held out once; no subject is silently skipped. Assertions
check subject separation, calibration/test identity, allowed sessions, row disjointness,
and all three classes in every **restricted S1-S2 training fold** before any model fits.
Held-out S2 is absent from training, calibration and evaluation arrays. S2 can still
serve as training data when that subject is not held out in another fold.

## Reuse, feature space and model

The runner imports 6.3 and its 6.2 module without invoking either CLI. It reuses
`collect`, `validate_features`, `feature_sets`, `fixed_parameters`, `build_pipeline`,
the 6.2 coverage gate, and the corrected 6.3 `predict_target` / `transform_target`.
The 6.4 split logic is separate because ordinary 6.2 LOSO trains on all sessions.

Feature definitions/order are unchanged: Spectral 620, Connectivity 350, Fusion 970.
Metadata/labels never enter X. One pipeline is fitted per dataset/held-out subject
on raw other-subject S1-S2 features and reused for both conditions:

```text
Training:     raw training -> train StandardScaler -> selector -> SVM
zero_cal:     raw target S3 -> train StandardScaler -> fitted selector -> fitted SVM
session1_cal: raw target S3 -> target z-score using held-out S1 statistics
                           -> fitted selector -> fitted SVM
```

Calibrated targets bypass the training scaler entirely. No scaler, selector or
classifier is refitted on any held-out rows. Population standard deviation uses
`ddof=0`; `std < 1e-10` becomes `1.0`. A one-window pilot calibration therefore
provides only an offset (`x - mu`), not a stable variance estimate. Zero calibration
computes no held-out statistics and receives only S3 features at prediction time.

The coordinate-space assumption follows 6.3: S1 target statistics substitute for
the training scaler at inference. It does not assume S1/S3 stationarity, matched
training/target distributions, or guaranteed improvement. Training and zero-cal
inference retain 6.2 behavior, but scores need not equal 6.2 because row splits differ.

Pipeline: `StandardScaler -> SelectKBest(f_classif, min(1200,n_features)) -> SVC(RBF)`.
All features are retained with the current counts. Fixed historical SVM_Tuned mapping:

| Dataset | Historical parameter row |
|---|---|
| Spectral | Spectral |
| Connectivity | GC+PDC_ROI |
| Fusion | GC+PDC_ROI+Spectral |

Parameters come from the same Phase 4 CSV as 6.2/6.3, with no fallback or retuning.
`gamma=scale` is estimated only by fitting the training fold. Label availability is
checked on training folds; evaluation labels are used only for scoring after prediction.
Ingestion still checks fixed trial-label consistency as dataset integrity validation;
this is not calibration selection or target-based model selection.

## Coverage and assumptions

Only validated passed 6.1 trials are accepted, latest passed reruns are deduplicated,
and incompatible extraction provenance is rejected by 6.2 ingestion. It validates
every retained full window against each trial's input shape, window IDs and sample
bounds. Expected trial count is 15/session; expected window count is **not assumed
constant** and comes from each validated trial report.

Final/default mode requires all 15 complete subjects, 45 passed trials each (675
trials total), including complete S1 and S3. The additional session gate checks all
expected trials per session. The coverage report records missing trials and windows
for all 45 configured subject/session combinations. No incomplete subject is skipped.

Pilot allows partial trials/windows with at least two subjects, S1 and S3 present for
every available subject, non-empty other-subject S1-S2 training, and all training
classes. S1 is required even for a zero-only pilot so the evaluated subject cohort
has the same session eligibility. Pilot calibration means all **available** passed
S1 windows, not necessarily a complete session. Held-out S2 remains excluded even
though final coverage requires it (it trains other folds). Audit success means input
inspection completed; it does not imply pilot/final eligibility.

## Metrics and output schema

Run directories are unique: `output/{audit,pilot,final}/TIMESTAMP/` and
`figures/{pilot,final}/TIMESTAMP/`. A completed experiment writes:

| File | Contents |
|---|---|
| `cross_session_per_subject.csv` | mode, protocol, dataset, model, subject_num/subject_id (held-out subject), condition, train_session_set, calibration_session, test_session, feature/window counts, accuracy, macro_f1, weighted_f1, confusion_matrix, gain_pp |
| `cross_session_summary.csv` | mode/protocol/dataset/condition and session identifiers; n_subjects, total n_test_windows; mean_* and std_* for accuracy, both F1 scores and gain_pp |
| `cross_session_predictions.csv` | original test-window metadata, condition/dataset/model/protocol/mode/session identifiers, y_true, y_pred |
| `cross_session_split_manifest.csv` | one row per input window/fold/condition, held_out_subject, window identity/bounds, role; no labels or features; shared across datasets |
| `coverage_report.json` | inherited 6.2 coverage, selected source paths/hashes, per-session coverage, pilot/final eligibility and rejection reason |
| `run_report.json` | status, split/transform definitions, parameters/mapping/hash, exact feature lists, code/config hashes, limitations |

`train_session_set` is the CSV string `1,2`; `calibration_session=0` means none and
`1` means S1; `test_session=3`. Counts are `n_train_windows`, `n_calibration_windows`,
`n_test_windows` in the per-subject table. The manifest role values are `train`,
`calibration`, `test`, `excluded_heldout_session2`, `excluded_other_session3`, and
`excluded_zero_cal_session1`; every input window has exactly one role per fold/condition.

Accuracy and F1 are proportions. F1 and confusion matrices use fixed labels [0,1,2]
(negative, neutral, positive), `zero_division=0`. Confusion matrices are JSON arrays,
actual class on rows, predicted class on columns.

```text
session1_cal gain_pp = 100 * (Acc_session1_cal - Acc_zero_cal)
zero_cal gain_pp = 0
```

Both conditions score exactly the same complete available S3 rows; gains are naturally
matched-row, with no change in sample composition. Summaries use equal subject weights,
mean and **sample SD (ddof=1)**, not pooled-window accuracy. SD is not a confidence
interval. The summary gain is the mean of paired per-subject gains.

Figures `cross_session_accuracy.png` and `cross_session_macro_f1.png` compare
zero_cal/session1_cal across requested datasets, with subject SD error bars. Default
plots contain all three feature sets. Pilot/final mode and primary protocol are labeled.
Audit writes only coverage/run JSON reports, no accuracy tables or plots.

## CLI and validation

From repository root, safe checks:

```powershell
python -B phase_6_wearable_evaluation/6.4_cross_session/code/test_cross_session.py
python -B phase_6_wearable_evaluation/6.4_cross_session/code/run_cross_session.py --audit-only
```

Experiments require a separate request and sufficient data:

```powershell
python -B phase_6_wearable_evaluation/6.4_cross_session/code/run_cross_session.py --pilot --datasets Spectral --conditions zero_cal session1_cal
python -B phase_6_wearable_evaluation/6.4_cross_session/code/run_cross_session.py --require-all-subjects --datasets Spectral Connectivity Fusion
```

Default is final coverage gating, primary protocol, all datasets, both conditions.
`--pilot`/`--require-all-subjects` are mutually exclusive. Requesting session1_cal alone
also includes zero_cal as the required gain anchor; zero_cal alone is supported.
Dependencies are those of 6.2 (numpy, pandas, scikit-learn, matplotlib).
Synthetic tests use temporary directories inside 6.4 and remove them automatically.

## Interpretation for the paper

Use “cross-session subject-independent evaluation”, “unlabeled Session-1 calibration”,
and “held-out Session-3 evaluation”. This is an offline dataset simulation, not evidence
of real-time deployment, longitudinal clinical validation, or wearable validation.
Session indices indicate within-subject ordering; different subjects' recordings do
not share calendar days or fixed day gaps. Repeated SEED trial/stimulus structure and
offline preprocessing remain limitations. Historical hyperparameters were tuned on
the same SEED cohort, so this is not independent/nested model selection.

The schema supports paper tables and paired subject-level comparisons with provenance,
but pilot/synthetic outputs must not be presented as final empirical results. Final
paper results require complete coverage and an explicitly requested real-data run.
