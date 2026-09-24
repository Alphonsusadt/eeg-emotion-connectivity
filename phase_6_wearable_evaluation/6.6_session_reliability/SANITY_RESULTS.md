# Phase 6.6 sanity results

Validated on 2026-09-24. **14 synthetic unit/sanity tests passed in 9.069 seconds**
(exit code 0). No real-data reliability analysis or classification was run.

Fixture: three subjects, three sessions, 81 input windows, 27 complete matched
subject/trial/window targets, 970 features, unequal target counts across subjects,
and one constant feature. Default union output has 970 unique feature rows,
2910 pairwise rows and separately labeled method/band ICC summaries. A separate
coverage-only fixture contains all 675 subject/session/trial combinations.

Verified:

- Perfect repeated session values produce ICC2/ICC3=1. A constant session offset
  preserves perfect consistency while lowering absolute agreement.
- Independent random session values have near-zero estimates; an anti-consistent
  fixture produces negative ICCs that are retained and descriptively labeled poor.
- Constant features return NaN with explicit undefined-denominator status; missing,
  non-finite or wrong-shaped ICC inputs reject. No zero filling or clipping of ICC.
- Session and target permutations preserve ICC, as does common affine rescaling.
- Explicit ANOVA mean squares and both ICC formulas agree within 1e-12 with an
  independent least-squares projection calculation on a small matrix.
- Matching uses all three keys and exactly three sessions. A missing S3 row removes
  its entire target and is recorded. Unequal per-session window counts use only the
  intersection, and duplicate target/session rows reject.
- Label changes and reversed feature-column order leave every reliability table
  unchanged. Guards prohibit classifier pipeline, parameter loading, LOSO/class
  checks or baseline model evaluation from being called during reliability analysis.
- All 970 features appear once despite Spectral/Connectivity/Fusion overlap. Every
  PDC band remains separately labeled, with 50 features each; spectral method/band
  groups contain 62 features each. The constant feature remains visible as undefined.
- Pearson/Spearman agree with SciPy, including rank ties and undefined constant
  vectors. All pairs use the same complete three-session targets, not extra pairwise rows.
- Subject-cluster bootstrap is seed-reproducible. Recorded draws prove every sampled
  subject contributes its complete unit block, duplicated intact on repeated draws.
  CI endpoints agree with independently collected replicate percentiles. Valid
  replicate counts and disabled/degenerate-CI statuses are correct.
- Family summaries include negative estimates and expose undefined counts; threshold
  proportions use the documented defined-feature denominator.
- Pilot with only two-session units rejects rather than silently changing ICC design.
  Noncontributing subjects are explicitly listed if other subjects have complete units.
  Complete final metadata passes; one missing S3 trial rejects. Empty audit input is supported.
- Optional subject/session mean reliability is separately named and matches direct
  aggregation. It uses the same matched-unit support and does not replace primary results.
- Five synthetic CSV outputs (including optional subject means) and all four figures
  are created under temporary 6.6 directories, then removed automatically.
- Audit-only computes no ICC, correlations, bootstrap or model, even if a requested
  bootstrap count is 1000; it only writes coverage/run reports and the matching manifest.

Only small synthetic bootstrap checks ran: 30 replicates on a two-feature fixture,
six on the 970-feature output fixture, and five on a constant fixture. No large or
real-data bootstrap was performed.

Commands completed successfully:

```text
python -B phase_6_wearable_evaluation/6.6_session_reliability/code/test_session_reliability.py
python -B phase_6_wearable_evaluation/6.6_session_reliability/code/run_session_reliability.py --audit-only --bootstrap 0 --seed 42
```

Current real-data audit: S01 / S1 (20131027) / trial 01 has three full windows.
S2/S3 are absent. There are **zero complete three-session targets**, three excluded
union targets, zero contributing/complete subjects, and 674 missing trials out of
675. Both pilot and final are ineligible. No missing session was replaced or inferred.

Reports: [coverage](output/audit/20260924T031313_185230Z/coverage_report.json),
[run](output/audit/20260924T031313_185230Z/run_report.json),
[matching manifest](output/audit/20260924T031313_185230Z/reliability_matched_unit_manifest.csv).

Phase 5 source was audited without executing notebooks: its ICC-like one-way subject
variance ratio has no repeated-session factor, clips negatives, fills missing values
and uses label-based PDC band selection. Phase 6.6 implements new, explicitly
session-based formulas and raw complete-case feature estimation. No previous phase
or historical results were edited. Schema is ready for TRON tables/figures, but final
empirical results and subject-cluster CIs await eligible data and a requested run.

Do not describe the old Phase 5 ICC as test–retest reliability. Phase 6.6 is the first analysis that explicitly treats the three SEED sessions as repeated measurements. Preserve this distinction in README and later paper text.
