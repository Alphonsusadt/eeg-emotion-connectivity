# Phase 6.3: Unlabeled calibration length evaluation

Target-only calibration of 60-second window features. This stage imports Phase 6.2
without running its CLI and reuses `collect`, `validate_features`, `feature_sets`,
`eligibility`, `checked_splits`, `fixed_parameters`, and `build_pipeline`.
All new files/results remain here. Phases 4, 5, 6.1 and 6.2 are unchanged.

## Exact protocol

Feature sets are unchanged: Spectral (620), Connectivity (350), Fusion (970).
Metadata and labels never enter X. LOSO holds out all rows of one subject; training
uses other subjects only. One fixed SVM pipeline is trained per dataset/fold and reused
for every condition. No new tuning or condition selection based on performance.

| CLI condition | Calibration rows of held-out subject | Evaluation rows | Oracle flag |
|---|---|---|---|
| `0` | None; compute no target statistics | All subject rows | False |
| `1`, `2`, `5`, `10` | First N windows ordered by session_index → trial → window_id | All remaining rows | False |
| `session1` | Every window with session_index=1 | Rows from later sessions | False |
| `full` | Every held-out window | All subject rows | True |

`full` means **full-subject oracle/reference normalization**, never a deployment baseline.
It is a transductive reference: evaluation features themselves contribute to its statistics.
Realistic primary conditions are 0, 1, 2, 5, and 10 minutes of unlabeled calibration.
Session1 is a longer onboarding comparison, not the full-subject oracle.

Durations count retained EEG windows: one window = one minute. They are not elapsed
wall-clock onboarding time across breaks, discarded trial tails, or session gaps.
`full`/`session1` duration is the actual number of included windows in minutes.

### Transformation order

1. Training subjects remain in raw feature space: **no subject-wise normalization**.
2. Fit `StandardScaler → SelectKBest(f_classif, min(1200,n_features)) → SVC(RBF)`
   only on training subjects, exactly as in 6.2. With ≤970 features the selector keeps
   all features. Historical fixed parameter mapping is identical to 6.2:
   Spectral → Spectral; Connectivity → GC+PDC_ROI; Fusion → GC+PDC_ROI+Spectral.
3. For target calibration >0, compute feature-wise `mu_cal` and population std
   (`ddof=0`) **only on raw calibration feature rows**. Replace `std < 1e-10` with 1.0.
   Transform evaluation features as `(x-mu_cal)/std_cal`. For 0, pass raw features unchanged.
4. Apply the already fitted training pipeline to those evaluation features. Neither
   scaler, selector nor SVM is fitted/updated on calibration or evaluation features.
5. Only after prediction, read evaluation labels for scoring. Selection accepts a
   metadata-only frame; statistics accept feature arrays without labels.

For one calibration window, every feature std is zero, so the rule becomes `x-mu_cal`
with denominator 1.0. This estimates an offset, not a reliable feature variance.
Target-only z-scoring can put target features on a different scale from raw training
features. This is the explicitly requested comparison; improvement is **not guaranteed**.
The full reference is not an upper bound and is not a reproduction of Phase 5's
normalization (which also normalized training subjects and used pandas sample std).

## Coverage, validity, and pilot behavior

Phase 6.2 logic accepts only validated `passed` 6.1 runs, deduplicates reruns, and
requires consistent schema/extraction provenance. Final/default mode requires all
15 subjects with 45 passed trials each, containing every full 60-second window.
Pilot accepts partial coverage with at least two subjects and all classes in training.
In a partial pilot, “first” means first **available** window, not proof of actual
complete onboarding chronology; missing trials are listed in coverage_report.

Every requested condition must have enough calibration rows and a non-empty evaluation
set for every held-out subject. No silent truncation, skipped folds, or scoring of limited
calibration rows. Unsupported conditions cause rejection **before any model fitting**.
Use `--conditions` to request feasible conditions for a pilot; `0` and `full` are always
included as gain anchors. All condition/plot ordering is `0,1,2,5,10,session1,full`.

Ingestion inherits 6.2's fixed trial-label consistency check; it checks dataset integrity,
not target-label-based calibration selection. LOSO validation checks class availability
on training folds only. Target calibration labels never determine row selection,
statistics, fitting, hyperparameters, or feature selection. Calibration selections are
exported without labels for auditability.

## Metrics and comparison with 6.2

Per subject/dataset/condition: accuracy, macro F1, weighted F1, confusion matrix
(actual rows, predicted columns; labels 0,1,2), calibration count/duration, evaluation
count, training count, and oracle flag. F1 uses fixed three-class labels and zero_division=0.

Requested metrics use each condition's own evaluation rows:

```text
gain_pp = 100 * (Acc_cal - Acc_0)
gap_to_full_pp = 100 * (Acc_full - Acc_cal)
recovery_pct = 100 * (Acc_cal - Acc_0) / (Acc_full - Acc_0)
```

Recovery is NaN when `Acc_full - Acc_0 <= 1e-10`, including negative gaps.
NaN is stored as an empty CSV field. Recovery is not clipped to 0–100: values outside
that interval can occur. Mean recovery is the mean of valid per-subject recoveries,
not a ratio of mean accuracies; `n_valid_*` records denominators of the summaries.

**Evaluation sets differ across conditions**: limited/session1 exclude calibration,
while 0/full score all rows. Therefore requested gains alone do not isolate adaptation
from changing sample composition. Additional `accuracy_0_same_rows`,
`accuracy_full_same_rows`, and `matched_gain_pp`, `matched_gap_to_full_pp`,
`matched_recovery_pct` recompute comparisons using identical evaluation rows for that
condition. Full remains transductive even in these matched comparisons. Different
limited conditions still use different row sets; do not overstate a causal dose response.

Summary is mean ± sample SD (ddof=1) across subjects, each subject weighted equally.
Window counts are audit information, not weighting factors. 0-minute predictions and
metrics are checked against the 6.2 implementation in synthetic tests. For later real
comparisons, match selected-run CSV hashes, feature lists, subject coverage and parameter
hashes; compare 6.3 condition 0 with the corresponding 6.2 outputs by window keys.
No old aggregate score is substituted for the current run's zero anchor.

Historical hyperparameters were tuned on the same SEED cohort: these results are not
independent/nested model-selection estimates. Features also inherit offline preprocessing;
this experiment does not establish causal acquisition or wearable real-time performance.

## CLI

From repository root:

```powershell
# Safe validation only; no real-data training
python -B phase_6_wearable_evaluation/6.3_calibration_length/code/test_calibration_length.py
python -B phase_6_wearable_evaluation/6.3_calibration_length/code/run_calibration_length.py --audit-only

# Run only when experiments are requested and inputs suffice
python -B phase_6_wearable_evaluation/6.3_calibration_length/code/run_calibration_length.py --pilot --datasets Spectral --conditions 0 1 2 full
python -B phase_6_wearable_evaluation/6.3_calibration_length/code/run_calibration_length.py --require-all-subjects --datasets Spectral Connectivity Fusion
```

`--pilot` and `--require-all-subjects` are mutually exclusive. Default is final gating;
default datasets are all three and default conditions are all seven. `--audit-only`
reports coverage and condition feasibility but never calls the evaluator. Audit success
means input inspection completed, **not** that final evaluation is eligible.
Dependencies: those of 6.2 (numpy, pandas, scikit-learn, matplotlib).

## Outputs

Unique `output/{audit,pilot,final}/TIMESTAMP/` and `figures/{pilot,final}/TIMESTAMP/`:

- `coverage_report.json`: 6.2 coverage/provenance plus condition availability per subject.
- `run_report.json`: status, calibration design, datasets, parameters/hashes, code hashes,
  feature names, oracle terminology, and interpretation limitations. Audit status is
  `audit_only`; only completed experiments have status `passed`.
- `calibration_per_subject.csv`, `calibration_summary.csv`, `calibration_predictions.csv`.
- `calibration_curve.csv`: summary ordered for plotting with valid-subject counts.
- `calibration_selection.csv`: subject/chronological keys selected for calibration, no labels.
- `accuracy_vs_calibration.png`, `macro_f1_vs_calibration.png`: main 0/1/2/5/10/full plots.
- Corresponding `*_all_conditions.png` includes session1 in prescribed order. Full is
  explicitly marked oracle on the x-axis. Plot error bars represent subject SD.

Audit-only produces reports, not accuracy tables or figures. Synthetic tests create
temporary output under this stage and remove it afterward; it is not empirical evidence.
