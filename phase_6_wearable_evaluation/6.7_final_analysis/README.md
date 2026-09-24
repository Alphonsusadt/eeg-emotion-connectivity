# Phase 6.7: Final analysis and paper-ready evidence integration

Phase 6.7 is an integration layer, not a new experiment. It reads, validates,
reconciles, summarizes, and visualizes already-completed **final** results from
Phases 6.2–6.6. It never extracts EEG features, fits a classifier, tunes an SVM,
runs reliability bootstrap, or performs streaming adaptation.

## Evidence boundary

Only the latest compatible run under `output/final/<run>/` is eligible when its
`run_report.json` has `status: passed`, `mode: final`, full 15-subject coverage is
proved by `coverage_report.json`, and every required result file exists. Pilot,
audit, synthetic, rejected, incomplete, and files outside `output/final` are never
fallback evidence. Multiple passed final runs with different scientific
fingerprints (feature schema, parameters, source inputs, protocol, normalization,
or dataset declaration) are rejected as ambiguous rather than silently selected.

The prerequisite gate passes only when 6.2, 6.3, 6.4, 6.5, and 6.6 all have an
eligible run. Cross-phase validation additionally requires:

- exactly 15 zero-calibration subjects per 6.2 dataset;
- consistent `Spectral`, `Connectivity`, and `Fusion` dataset names;
- unique subject/condition and feature rows and finite performance metrics;
- 6.3 condition `0` exactly equal to 6.2 by dataset, subject, and metric;
- 6.5 `zero_cal`/`fixed_session1` exactly equal to 6.4
  `zero_cal`/`session1_cal` on the shared protocol;
- explicit valid/undefined ICC status, with finite ICC values when defined and
  blank values when undefined.

The default numeric tolerance is `1e-10`. Exact mismatching keys are reported and
final integration is rejected.

## Paper scope and research questions

The final summary is organized around:

1. RQ1 — window-level unseen-subject zero-calibration performance (6.2).
2. RQ2 — unlabeled calibration length using 0/1/2/5/10 minutes, with `full` only
   as a transductive oracle/reference and matched-row metrics as headlines (6.3).
3. RQ3 — Session-1 calibration versus zero calibration on held-out Session 3 (6.4).
4. RQ4 — zero, fixed Session-1, causal running, and causal EMA normalization (6.5).
5. RQ5 — ICC(2,1), ICC(3,1), all method/band families, and pairwise session
   associations without selecting only high-ICC families (6.6).

Planned performance comparisons use paired subject IDs and two-sided Wilcoxon
signed-rank tests. The limited set of planned comparisons is Holm-corrected as one
family. Raw and corrected p-values, mean and median paired differences in percentage
points, and exact subject IDs are saved. A p-value never replaces effect size or
licenses a stronger claim. Negative and null outcomes are kept.

## Phase 5 is historical, not a comparator on the same scale

`historical_phase5_reference.csv` separately records the requested Phase 5 results:

| Dataset/model | Accuracy mean ± SD |
|---|---:|
| Spectral / Voting Soft | 74.22 ± 9.27% |
| GC / SVM Tuned | 60.00 ± 16.35% |
| PDC Theta / Voting Soft | 60.15 ± 11.06% |
| GC+PDC_ROI+Spectral / Voting Soft | 75.70 ± 9.99% |
| Combined / SVM Tuned | 75.26 ± 8.52% |

Every row is labeled `analysis_level=trial`,
`normalization=full_subject_unlabeled_reference`, and
`protocol=historical_phase5_loso`. Phase 5 used trial-level features and
full held-out-subject unlabeled normalization. Phase 6 uses 60-s window-level
features and evaluates zero/limited calibration, cross-session transfer, causal
adaptation, and session reliability. They are not identical or apples-to-apples;
no direct improvement claim is computed.

## CLI

From the repository root:

```powershell
# Safe now: inspect availability and schemas; never generate final claims
python -B phase_6_wearable_evaluation/6.7_final_analysis/code/run_final_analysis.py --audit-only

# Run only after all five eligible final upstream runs exist
python -B phase_6_wearable_evaluation/6.7_final_analysis/code/run_final_analysis.py --require-complete --tolerance 1e-10

# Synthetic unit tests only
python -B phase_6_wearable_evaluation/6.7_final_analysis/code/test_final_analysis.py
```

`--audit-only` writes timestamped `prerequisite_report.json`,
`upstream_provenance.json`, and `run_report.json`. It exits successfully even when
final evidence is missing, but clearly lists missing/rejected phases and never writes
paper tables, figures, or claims. Normal final mode is fail-closed; `--require-complete`
makes that intended gate explicit.

## Final outputs

Successful final integration writes under `output/final/<timestamp>/`:

- `final_baseline_table.csv`
- `final_calibration_table.csv`
- `final_cross_session_table.csv`
- `final_streaming_table.csv`
- `final_reliability_table.csv`
- `planned_pairwise_tests.csv`
- `historical_phase5_reference.csv`
- `upstream_provenance.json` (selected paths and code/config/schema/parameter/source hashes)
- `paper_results_summary.md`
- `run_report.json` and `prerequisite_report.json`

Paper figures are written under `figures/final/<timestamp>/`:
`fig_main_baseline.png`, `fig_calibration_curve.png`, `fig_cross_session.png`,
`fig_streaming_adaptation.png`, and `fig_reliability.png`. Classification axes use
0–1. Reliability retains negative ICC values and shows ICC(2,1) separately from
ICC(3,1); error bars there are feature IQR, not confidence intervals.

The claim-safe wording is limited to subject-independent, cross-session, unlabeled
adaptation, causal streaming normalization, offline deployment-oriented evaluation,
and “toward wearable monitoring.” It does not claim wearable readiness, clinical
validation, real-time validation, or connectivity superiority unless the actual
final evidence supports a narrower descriptive statement.

Phase 6.7 must never convert a pilot, audit, synthetic, or incomplete result into a final paper result. Missing evidence must remain missing. Negative or null findings must be preserved, not optimized away.

