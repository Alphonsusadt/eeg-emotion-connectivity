# Pre-paper execution readiness status

Status date: 2026-09-24. Fresh equivalence and the representative six-trial pilot
passed, but real-data paper evidence is not yet complete and is not frozen.

## Complete implementation

- Frozen 60-s, non-overlapping 6.1 scientific feature protocol remains intact.
- Resumable 675-trial batch enumeration with current-provenance validation, duplicate
  suppression, missing/failed/incompatible classification, filtering, dry-run, and audit-only.
- Conservative process-level parallelism with one numerical thread per worker.
- Fail-closed serial/worker feature, matrix, and metadata equivalence gate.
- Per-stage timing (`load/preprocess`, spectral, GC, PDC, ROI, save) and aggregate batch timing.
- Read-only 675-trial coverage dashboard with subject/session summaries.
- GC density/p-value/FDR audit and exact reproduction check against the active implementation.
- Phase 6.2–6.7 tests, leakage guards, provenance gates, and integration schemas.
- Gated final provenance manifest builder and explicit paper-freeze checklist.

## Real-data work pending

- Full 675-trial extraction.
- Final 6.2, 6.3, 6.4, 6.5, 6.6 (1000 bootstrap, seed 42), and 6.7 in order.
- Final provenance freeze and read-only evidence archive.

## Current coverage and compute estimate

Dashboard result: 6/675 current-compatible passed, 0 failed, 669 missing, and 0
incompatible after selecting current runs. Fresh serial/worker equivalence passed at
`1e-10`; the selected six-trial subset is 6/6 passed with 18 windows.

The selected pilot mean/median trial runtime was 1436.9/1178.1 seconds. A naive mean-
based serial projection is 269.4 hours. The observed five-new-trial two-worker batch
throughput was 4.16 trials/hour, implying a rough 162.2-hour (6.76-day) full projection.
This is not linear-scaling evidence: two timings crossed user-turn interruptions and
the queue tail used only one worker. An early snapshot saw ~199–201 MiB RSS per child,
7.6 GiB RAM available, and no extraction OOM/timeout. Continuous peak/thermal telemetry
was unavailable. Keep 2 workers; do not automatically increase to 3.

## Next commands

```powershell
# Inspect pilot without extraction
python -B phase_6_wearable_evaluation/6.1_windowed_feature_extraction/code/run_batch_extraction.py --subjects 1 2 --sessions 1 --trials 1 2 3 --workers 2 --resume --dry-run

# Explicit pilot after parallel-equivalence report passes
python -B phase_6_wearable_evaluation/6.1_windowed_feature_extraction/code/run_batch_extraction.py --subjects 1 2 --sessions 1 --trials 1 2 3 --workers 2 --resume

# Full extraction, only after pilot review
python -B phase_6_wearable_evaluation/6.1_windowed_feature_extraction/code/run_batch_extraction.py --workers 2 --resume
```

The complete downstream command sequence is in `FINAL_EXECUTION_PLAN.md`.

## Canonical paper evidence after freeze

The canonical result directory will be the selected passed final 6.7 run containing
the five final CSV tables, planned paired tests, historical Phase 5 reference,
`paper_results_summary.md`, `upstream_provenance.json`, `prerequisite_report.json`,
and `run_report.json`; its paired `figures/final/<timestamp>/` directory contains the
five paper figures. `final_provenance_manifest.json` binds these to the repository,
configuration, phase code, 6.1 selected runs, feature schema, and tuning parameters.

No paper claim should be drafted from current pilot/audit/synthetic evidence.
