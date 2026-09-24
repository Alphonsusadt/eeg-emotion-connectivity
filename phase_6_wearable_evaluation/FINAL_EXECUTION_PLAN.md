# Phase 6 final execution plan

This is an operator runbook, not an auto-launch script. Run one gate at a time,
inspect its report, and continue only after it passes. No command below is executed
automatically by this document.

## 0. Lock environment and inspect

Record the repository commit and Python/package environment. Do not modify Phase 4/5
or the scientific protocol in 6.2–6.7 after extraction begins.

```powershell
git rev-parse HEAD
python --version
python -B phase_6_wearable_evaluation/6.1_windowed_feature_extraction/code/test_windows.py
python -B phase_6_wearable_evaluation/6.1_windowed_feature_extraction/code/check_extraction_coverage.py
```

## 1. Parallel-equivalence gate and representative pilot

First produce two fresh runs of the same trial using the single-trial code path and
the restricted one-trial/one-child process probe. Compare them before enabling more
than one concurrent worker:

```powershell
python -B phase_6_wearable_evaluation/6.1_windowed_feature_extraction/code/extract_windows.py --subject 1 --session-index 1 --trial 1
python -B phase_6_wearable_evaluation/6.1_windowed_feature_extraction/code/run_batch_extraction.py --subjects 1 --sessions 1 --trials 1 --worker-probe
python -B phase_6_wearable_evaluation/6.1_windowed_feature_extraction/code/check_parallel_equivalence.py --serial-run PATH_SERIAL --worker-run PATH_WORKER --tolerance 1e-10
```

Then inspect the six-trial plan (S01/S02, Session 1, trials 1–3; all emotion classes):

```powershell
python -B phase_6_wearable_evaluation/6.1_windowed_feature_extraction/code/run_batch_extraction.py --subjects 1 2 --sessions 1 --trials 1 2 3 --workers 2 --resume --dry-run
```

Only after equivalence passes, explicitly run the pilot by removing `--dry-run`.
After completion, run the coverage dashboard and 6.2 pilot eligibility audit. Do not
promote pilot scores to final evidence.

```powershell
python -B phase_6_wearable_evaluation/6.2_windowed_baseline/code/run_windowed_loso.py --audit-only
python -B phase_6_wearable_evaluation/6.2_windowed_baseline/code/run_windowed_loso.py --pilot
```

## 2. Full 6.1 extraction

Inspect first, then explicitly remove `--dry-run` when compute resources are ready:

```powershell
python -B phase_6_wearable_evaluation/6.1_windowed_feature_extraction/code/run_batch_extraction.py --workers 2 --resume --dry-run
python -B phase_6_wearable_evaluation/6.1_windowed_feature_extraction/code/run_batch_extraction.py --workers 2 --resume
python -B phase_6_wearable_evaluation/6.1_windowed_feature_extraction/code/check_extraction_coverage.py
```

Start with 2 workers. Use 3–4 only if the pilot demonstrates adequate RAM, no paging,
stable thermals, and near-linear throughput. Every numerical worker is constrained to
one BLAS/OpenMP thread. Interruptions are recoverable: rerun the same command and valid
passed trials are skipped.

Do not proceed until coverage is exactly 675/675 current-compatible passed trials.

## 3. Final execution order (6.2–6.7)

Run tests/audits first, then each final command in this exact order. Stop on any
rejection or provenance mismatch.

```powershell
# 6.2 zero-calibration windowed LOSO
python -B phase_6_wearable_evaluation/6.2_windowed_baseline/code/test_windowed_loso.py
python -B phase_6_wearable_evaluation/6.2_windowed_baseline/code/run_windowed_loso.py --audit-only
python -B phase_6_wearable_evaluation/6.2_windowed_baseline/code/run_windowed_loso.py --require-all-subjects

# 6.3 calibration length
python -B phase_6_wearable_evaluation/6.3_calibration_length/code/test_calibration_length.py
python -B phase_6_wearable_evaluation/6.3_calibration_length/code/run_calibration_length.py --audit-only
python -B phase_6_wearable_evaluation/6.3_calibration_length/code/run_calibration_length.py --require-all-subjects --datasets Spectral Connectivity Fusion

# 6.4 cross-session
python -B phase_6_wearable_evaluation/6.4_cross_session/code/test_cross_session.py
python -B phase_6_wearable_evaluation/6.4_cross_session/code/run_cross_session.py --audit-only
python -B phase_6_wearable_evaluation/6.4_cross_session/code/run_cross_session.py --require-all-subjects --datasets Spectral Connectivity Fusion

# 6.5 causal streaming normalization; alpha remains fixed, not tuned
python -B phase_6_wearable_evaluation/6.5_streaming_normalization/code/test_streaming_normalization.py
python -B phase_6_wearable_evaluation/6.5_streaming_normalization/code/run_streaming_normalization.py --audit-only
python -B phase_6_wearable_evaluation/6.5_streaming_normalization/code/run_streaming_normalization.py --require-all-subjects --datasets Spectral Connectivity Fusion --ema-alpha 0.1

# 6.6 session reliability; paper run uses subject-cluster bootstrap
python -B phase_6_wearable_evaluation/6.6_session_reliability/code/test_session_reliability.py
python -B phase_6_wearable_evaluation/6.6_session_reliability/code/run_session_reliability.py --audit-only --bootstrap 0 --seed 42
python -B phase_6_wearable_evaluation/6.6_session_reliability/code/run_session_reliability.py --require-all-subjects --bootstrap 1000 --seed 42 --datasets Spectral Connectivity Fusion

# 6.7 integration only
python -B phase_6_wearable_evaluation/6.7_final_analysis/code/test_final_analysis.py
python -B phase_6_wearable_evaluation/6.7_final_analysis/code/run_final_analysis.py --audit-only
python -B phase_6_wearable_evaluation/6.7_final_analysis/code/run_final_analysis.py --require-complete --tolerance 1e-10
```

## 4. Freeze provenance

Only after the checklist passes:

```powershell
python -B phase_6_wearable_evaluation/freeze_final_provenance.py
```

The command is fail-closed. It creates `final_provenance_manifest.json` only when
675/675 current-compatible 6.1 trials and passed final 6.2–6.7 runs exist. The schema
contains repository commit SHA, config hash, code hashes per phase, every selected
6.1 run/report/feature hash, final run paths, feature schema hash, historical tuning
parameter hash, and explicit freeze assertions.
