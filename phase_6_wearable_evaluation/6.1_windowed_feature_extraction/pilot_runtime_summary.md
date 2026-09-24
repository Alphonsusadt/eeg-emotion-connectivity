# Fresh equivalence and representative pilot runtime summary

Run date: 2026-09-24. This is engineering evidence, not paper evidence.

## Go/no-go result

**GO for the tested two-worker execution strategy.** Fresh serial and one-child
worker results for S01/Session 1/Trial 1 passed strict equivalence at absolute
tolerance `1e-10`. Feature CSV and every compressed matrix file had identical SHA256,
in addition to value-level comparison. Timing fields and paths were excluded from
scientific comparison as designed.

- Serial: `output/subject_01/session_20131027/trial_01/run_20260924T080534_899861Z`
- Worker: `output/subject_01/session_20131027/trial_01/run_20260924T083534_851672Z`
- Gate: `parallel_equivalence_report.json` (`status=passed`)

The comparison covered provenance/metadata, three windows, feature names/order,
970 feature values per window, boundaries, labels, GC raw/p-values/thresholded,
and all delta/theta/alpha/beta/gamma/broadband PDC matrices.

## Pilot coverage

The selected pilot subset is exactly S01/S02, Session 1, trials 1–3. Trial labels
are positive, neutral, and negative, mapping to classes 2, 1, and 0. All six trials
are current-compatible `passed`; each has three windows and 970 finite features per
window. Totals: 18 windows and 18 feature rows. Missing=0, failed=0, incompatible=0
within the selected subset. Global coverage is only 6/675 (0.89%): 669 remain missing.

| Subject | Trial | Windows | Selected run |
|---:|---:|---:|---|
| 1 | 1 | 3 | `run_20260924T083534_851672Z` |
| 1 | 2 | 3 | `run_20260924T090812_867592Z` |
| 1 | 3 | 3 | `run_20260924T090812_867592Z` |
| 2 | 1 | 3 | `run_20260924T094426_698047Z` |
| 2 | 2 | 3 | `run_20260924T094432_933882Z` |
| 2 | 3 | 3 | `run_20260924T100024_210313Z` |

## Runtime

For the six selected pilot runs, total runtime mean/median/min/max was
1436.9 / 1178.1 / 953.1 / 2180.4 seconds per trial. Mean component times were
1415.9 s GC, 20.5 s PDC, 0.10 s spectral, 0.018 s ROI, and 0.184 s save. GC remains
the dominant cost. The two S01 trial timings include two user-turn interruptions
and are not clean performance benchmarks.

The five-trial two-worker batch wall interval was 4326.2 seconds, or 4.16 completed
new trials/hour. A naive projection from the six selected per-trial mean is 269.4
serial hours for 675 trials. Applying the observed batch wall throughput gives
162.2 hours (6.76 days) for 675 trials. This is a rough operational projection,
not a claim of linear scaling: the batch contained only five newly extracted trials,
the final queue tail used one worker, and interruptions contaminated part of the run.

## Resource behavior

Two child workers overlapped successfully. An early snapshot observed approximately
199–201 MiB RSS per worker and 145 MiB for the parent, with 7.6 GiB system RAM
available, 61.4% RAM use, and 1.26 GiB pagefile use. No extraction failure, timeout,
or observed out-of-memory event occurred. Reliable continuous peak RAM, page-fault,
CPU-utilization, and thermal telemetry was not available; no peak value is inferred.

Recommendation: keep **2 workers** for the full run. Do not test or use 3 workers
until a separately monitored resource test is justified; this pilot does not show a
need for more concurrency.

## GC density and 6.2 readiness

Fresh GC audit passed with mean thresholded density `0.904812`, identical to the
historical audit (~0.9048). The active threshold/FDR behavior was not changed.

Phase 6.2 audit found subjects `[1,2]`, 18 windows, and no complete final subject.
Read-only eligibility/split validation passed: two LOSO folds and all three classes
are present in each training fold. A Spectral-only engineering pilot was not run
because the existing 6.2 CLI has no `--datasets` option; invoking `--pilot` would run
Spectral, Connectivity, and Fusion. Eligibility logic and Phase 6.2 were not changed.

