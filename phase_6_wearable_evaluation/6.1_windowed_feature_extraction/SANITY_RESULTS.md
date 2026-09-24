# Sanity test S01 / Session 1 / Trial 1

Status: PASSED. Input: 62 x 23,501 samples, 100 Hz.

Three non-overlapping 60-second windows; 5,501 tail samples dropped.
Each window: 970 finite features + 12 metadata columns; nine finite 62 x 62 matrices.

```text
 window_id  start_seconds  stop_seconds  gc_density  pdc_order  elapsed_seconds
         1            0.0          60.0    0.904812         12       458.804994
         2           60.0         120.0    0.902433         10       496.082872
         3          120.0         180.0    0.907192         10       495.176884
```

Total extraction time: 1450.1 seconds.

Artifacts: [sanity_report.json](output/subject_01/session_20131027/trial_01/run_20260923T163916_502869Z/sanity_report.json).

Boundary/tail and directed ROI tests: 2 passed. Persisted CSV/matrix checks passed.
Baseline source hashes verified unchanged. No classifier or full-dataset run performed.

## Pre-paper execution-readiness audit (2026-09-24)

- Eleven synthetic unit/readiness tests passed in 0.864 seconds.
- Verified boundary/direction behavior and a lightweight end-to-end single-trial
  extraction retaining 970 features, matrices, labels, and unique-run semantics.
- Verified completed-run skip, resumable failed/interrupted run, duplicate suppression,
  incompatible provenance rejection, exact 675-trial enumeration, and extraction-free
  dry-run/audit-only behavior.
- Verified strict serial/worker artifact comparison on feature CSV, GC/PDC matrices,
  stable manifest fields, metadata, and provenance at tolerance `1e-10`.
- Parallel execution remains disabled until a current real-trial equivalence report passes.
- Current-compatible dashboard: 0/675 passed, 0 failed, 674 missing, 1 incompatible.
  The previous S01/S1/T1 sanity result predates timing/extractor provenance instrumentation;
  it is retained for audit but is not accepted as current final extraction evidence.
- Read-only GC audit reproduced every stored thresholded matrix exactly. Across the
  three sanity windows, mean thresholded density is 0.904812. The active minimum-p-
  across-lags then BH-FDR behavior was preserved; no scientific threshold changed.
- No representative pilot or full extraction was run.
