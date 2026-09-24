# Phase 6.7 sanity results

Validated on 2026-09-24 during pre-paper execution preparation.

- All 13 synthetic unit/integration tests passed in 7.576 seconds.
- Tests covered latest compatible final-run selection; rejection of pilot/audit/synthetic,
  missing phases, ambiguous provenance, duplicate rows, non-finite metrics, and 6.4/6.5
  mismatch; exact table preservation; no experiment/model calls; matched-subject tests;
  6.3 matched-row gains; Holm correction; null/negative wording; and all final artifacts.
- `run_final_analysis.py --audit-only` completed successfully in approximately 2.6 seconds.
- Audit selected no final upstream runs. Missing passed-final phases: 6.2, 6.3, 6.4,
  6.5, and 6.6.
- Audit wrote only prerequisite/provenance/run reports. It generated no paper table,
  figure, statistical claim, or final result.
- The integration layer is ready to accept real final results after 6.1 reaches
  675/675 validated coverage and final 6.2–6.6 execute in order.

No pilot, audit, synthetic, or incomplete evidence was promoted to a paper result.
