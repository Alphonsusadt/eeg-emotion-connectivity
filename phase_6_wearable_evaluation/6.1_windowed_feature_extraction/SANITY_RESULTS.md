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
