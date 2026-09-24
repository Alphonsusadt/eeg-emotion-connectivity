# Pre-paper execution readiness status

Status date: 2026-09-24. The implementation is ready for controlled pilot execution,
but real-data paper evidence is not yet complete and is not frozen.

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

- Fresh serial/process-worker equivalence runs using the current instrumented hashes.
- Six-trial representative pilot: S01/S02, Session 1, trials 1–3.
- Full 675-trial extraction.
- Final 6.2, 6.3, 6.4, 6.5, 6.6 (1000 bootstrap, seed 42), and 6.7 in order.
- Final provenance freeze and read-only evidence archive.

## Current coverage and compute estimate

Dashboard result: 0/675 current-compatible passed, 0 failed, 674 missing, and 1
incompatible historical sanity run. The old run remains useful for runtime/density
audit but is not silently accepted after extractor provenance instrumentation.

Observed runtime is 1,450.1 seconds per representative three-window trial. A simple
constant-runtime projection is about 271.9 serial hours (11.3 days) for 675 trials.
Idealized compute-only projections are about 136.0 hours with 2 workers and 68.0 hours
with 4 workers; real wall time will be longer because trial/window counts, contention,
I/O, RAM, and thermal limits vary. This host reports 8 physical/16 logical CPU and
about 19.7 GiB RAM. Start with 2 workers after equivalence passes; treat 3 as the
pilot-tested ceiling unless memory measurements justify otherwise. The 4-worker
number is a scaling illustration, not the recommendation for this host.

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
