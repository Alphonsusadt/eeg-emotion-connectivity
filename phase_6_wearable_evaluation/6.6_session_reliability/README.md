# Phase 6.6: Session reliability / test-retest stability

Raw-feature **test-retest reliability** and **cross-session feature stability** of
validated 60-second SEED windows. This stage estimates **absolute-agreement ICC**
and **consistency ICC**, independently of classifier performance. It never trains
a classifier, reads SVM tuning parameters, uses predictions, or selects features
by emotion labels/accuracy. All new files and results remain in this directory.

## Audit of historical Phase 5 ICC

Before implementation, the source cells (without executing them) were inspected in
[5.5_icc_analysis.ipynb](../../phase_5_loso/5.5_icc_analysis/code/5.5_icc_analysis.ipynb)
and the duplicate implementation in
[phase_5_loso.ipynb](../../phase_5_loso/code/phase_5_loso.ipynb).

The old `compute_icc(values, groups)` groups only by `subject_num`. It computes
between-subject and within-subject mean squares, using the first subject's row count
as the common repeats count. Its ratio is
`(MS_between-MS_within)/(MS_between+(n_per_group-1)*MS_within)`.
There is no session column mean square, target-by-session residual, or matching of
the same trial/window position across sessions. Although labeled ICC(2,1) in the
notebook, this is a one-way grouped variance ratio under its balance assumption,
not the two-way session absolute-agreement estimator implemented here.

Further differences: Phase 5 returns 1 when within-group variance is zero, clips
negative estimates to zero, fills feature missing values with zero, excludes
near-constant features, and chooses one PDC band using label-based ANOVA scores.
The reliability loop operates on its historical raw trial-level features, despite
also constructing normalized classifier datasets earlier in the notebook.
None of those estimation/selection rules are reused in 6.6. The old files and
results are left untouched; their between-/within-subject analysis must not be
presented as session test-retest reliability.

## Input, units and matching

Phase 6.2 is imported only to reuse `collect`, `validate_features`, `feature_sets`,
the coverage gate, config and source hashing. Its CLI, LOSO splits, feature selection,
class-availability checks, classifier pipeline and parameter loader are never called.

Primary repeated-measure unit:

```text
target i = subject_num x trial x window_id
measurements j = session_index 1, 2, 3
feature matrix = n complete matched targets x 3 sessions
```

Each target must have exactly one row in each session. Join keys are strictly
`subject_num + trial + window_id`; no matching across subjects, trials, or window IDs.
Duplicate target/session rows are rejected. Different trial window counts are handled
by taking the intersection of available window IDs across all three sessions.
Missing rows are excluded as whole targets, recorded in the manifest, and never
interpolated, duplicated or replaced with zeros. All features and all pairwise
correlations use this same complete-case target set. Pairwise association does not
add extra two-session-only rows.

A window ID denotes the same retained segment position within the repeated trial,
not an identical physiological realization. Repeated SEED stimulus/trial structure
means reliability includes repeat-stimulus context. It is not evidence about
naturalistic long-term monitoring. Missing-window complete-case selection may alter
the retained sample composition; exclusions must accompany paper results.

Session indices map to each subject's dates through config, as validated during 6.2
ingestion. Indices are ordered occasions, not common calendar days across subjects.
The two-way model distinguishes a target effect, an occasion-column effect and
residual target-by-session variation. It does not separately identify all nested
subject/trial effects or subject-specific day trends.

## Raw features and family definitions

Primary input is raw extracted 6.1 feature values. There is no subject-wise z-score,
train StandardScaler, calibration or all-session normalization. Centering by the
grand mean inside ANOVA sums is arithmetic, not a normalization preprocessing step.
Finite input/schema validation is inherited; malformed passed runs are rejected,
not repaired or silently skipped. Ingestion checks trial labels for data integrity,
but the reliability computation never uses emotion labels.

Feature sets are unchanged: Spectral 620, Connectivity 350, Fusion 970. CLI datasets
select the **unique union** of named features. Requesting all three therefore yields
970 primary feature rows, not 1940 duplicated rows. Fusion is membership metadata,
not a new family to average with its components. All bands/features are retained,
including constant, low and negative-reliability features.

| Family | Method | Bands | Features per band/method |
|---|---|---|---:|
| Spectral | PSD_power (`spec_power_*`) | delta, theta, alpha, beta, gamma | 62 each |
| Spectral | Differential_entropy (`spec_de_*`) | delta, theta, alpha, beta, gamma | 62 each |
| Connectivity | GC_ROI | not_applicable | 50 |
| Connectivity | PDC_ROI | delta, theta, alpha, beta, gamma, broadband | 50 each |

Spectral method-wide summaries are additionally labeled `spectral_method_all_bands`.
PDC bands are always separate; no best-band selection or pooled PDC summary occurs.
GC's absent band label does not assert equivalence to any PDC band.

Phase 6.1 transposes PDC to source -> target before ROI aggregation. 6.6 uses this
orientation unchanged. It must not be equated with historical Phase 5 PDC orientation
or historical trial-level features without explicitly reporting that difference.

## Explicit two-way ANOVA ICC

For one feature, let x_ij be target i in session j, n matched targets and k=3 sessions.
Let r_i, c_j and g be the row, column and grand means:

```text
MSR = k * sum_i (r_i - g)^2 / (n - 1)
MSC = n * sum_j (c_j - g)^2 / (k - 1)
MSE = sum_ij (x_ij - r_i - c_j + g)^2 / ((n - 1)*(k - 1))

ICC(2,1) = (MSR - MSE) / (MSR + (k-1)*MSE + k*(MSC-MSE)/n)
ICC(3,1) = (MSR - MSE) / (MSR + (k-1)*MSE)
```

ICC(2,1) is two-way random-effects, absolute agreement, single measurement; ICC(3,1)
is two-way mixed-effects, consistency, single measurement. The formulas agree with
the explicit ANOVA estimators in the [official Pingouin source](https://github.com/raphaelvallat/pingouin/blob/master/src/pingouin/reliability.py).
The terminology follows [Shrout and Fleiss (1979)](https://pubmed.ncbi.nlm.nih.gov/18839484/).
No black-box ICC library is used or installed. A test independently derives mean
squares from least-squares projections of target/session dummy-variable models.

Systematic additive session offsets reduce absolute agreement while leaving perfect
consistency intact. ICC2 treats occasions as random under the specified model;
ICC3 treats these three occasion levels as fixed. The observed SEED sessions were
not randomly sampled calendar days, so broad generalization across days is an
assumption, not something this design establishes.

Targets are nested within subjects/trials; primary point estimates weight each
matched target equally. Subjects with more retained matched units contribute more
targets. A stable subject or stimulus effect can elevate ICC; this is not a pure
within-subject trait measure. Subject-cluster CIs below address dependence within
subjects without changing this point-estimate definition. No independent-window
ANOVA p-values or analytic independent-target confidence intervals are reported.

At least two targets and exactly three sessions are required. Non-finite/missing
feature values are rejected. Denominators must be finite and greater than
`64 * float64_epsilon * max(MSR,MSC,MSE)` per feature; zero or numerically
unidentified denominators return NaN with `undefined_denominator`. Entirely constant
features thus remain in the output with undefined ICC, rather than returning 1.
Finite negative estimates are retained without clipping to [0,1]. Mean squares and
explicit status columns are saved for inspection.

## Secondary pairwise association and subject means

Pearson r and Spearman rho are computed for S1-S2, S1-S3 and S2-S3 on the **same
three-session complete targets**. Spearman uses average ranks for ties. Constant
session vectors produce NaN and an explicit status. These are pairwise
association/stability measures, not reliability equivalent to ICC.

Optional `--subject-session-means` creates only the separate
`subject_session_mean_reliability.csv`. It averages each feature over the retained
complete-case trial/windows within each subject/session, then applies the same
three-session ICC formulas with subject as target. The common matched-unit support
avoids comparing session means built from different trial/window mixtures. This
is not a mean over unmatched available rows. `n_targets` becomes the number of
subjects; `n_matched_units` still reports the source target count. It does not
replace primary estimates or enter primary figures/family summaries.

## Subject-cluster bootstrap CIs

`--bootstrap 0` is the safe default in every mode. For a separately requested final
paper run use `--bootstrap 1000 --seed 42`; the code never automatically launches
a large bootstrap. Audit-only executes zero replicates regardless of the flag.

Each replicate samples S contributing subjects with replacement from S observed
contributing subjects. Every selection includes **all** matched trial/window units
and their three-session values; selecting the same subject twice duplicates its
whole block. Windows are never sampled independently and duplicate subject draws
are not deduplicated. The same draw is applied to every feature. A fixed NumPy RNG
seed gives reproducible draws; computation is vectorized across unique features.
Only one resampled cube and the per-feature ICC replicates are retained, not all
bootstrap input cubes. Runtime scales with matched units, features and replicates.

CI endpoints are the 2.5th/97.5th percentiles of finite bootstrap estimates (95%
percentile intervals), separately for ICC2 and ICC3, without clipping. Undefined
replicates are counted and excluded; fewer than two valid replicates leaves CI
undefined. The valid count and CI status are exported so degenerate resampling is
visible. This minimum is a computational rule, not a claim that two replicates
provide adequate uncertainty estimation. Disabled bootstrap produces blank CIs
with status `disabled`, never a fabricated interval.

The optional subject-mean analysis also resamples whole subjects. Bootstrap uncertainty
is conditional on the observed three occasions and repeated-stimulus structure;
sessions/stimuli are not resampled. Small pilot subject counts and missingness limit
inference; even 1000 replicates cannot compensate for too few independent subjects.

## Coverage gates and audits

Validated passed 6.1 runs, deduplication, provenance consistency and full-window
integrity come from 6.2. Expected full windows per trial derive from its validated
input shape, not an assumed universal trial duration.

- Final requires all 15 complete subjects, three complete sessions and 45 trials per
  subject (675 trials total), plus at least one matched target for every subject/trial.
- Pilot permits partial coverage with at least two available subjects, each having
  at least two sessions. Primary estimation additionally requires complete three-session
  units from at least two contributing subjects. A subject with S3 missing has zero
  complete units and is explicitly listed as noncontributing, not included in a
  two-session substitute ICC. If fewer than two contributors remain, the run rejects.
- Audit lists matching counts for all 15 configured subjects, missing sessions/trials,
  incomplete unit IDs, unmatched observed window-row counts, contributing subjects,
  and pilot/final eligibility reasons. The label-free manifest includes every observed
  union target with presence flags and an inclusion/exclusion reason. A trial missing
  in every session has no known window IDs; it is recorded in coverage, not fabricated
  as window rows. Audit success means inspection completed, not analysis eligibility.

No label distribution or classifier feasibility requirement applies to reliability.
All inference uses raw measurements, not `y_true`, `y_pred` or accuracy.

## Tables, summaries and figures

Unique directories: `output/{audit,pilot,final}/TIMESTAMP/` and
`figures/{pilot,final}/TIMESTAMP/` prevent overwriting historical outputs.

| Output | Contents |
|---|---|
| `reliability_per_feature.csv` | one row/unique feature; family/method/band and dataset membership; analysis/mode/raw marker; n_targets, n_matched_units, n_subjects, n_sessions; MSR/MSC/MSE; ICC2_1/ICC3_1, statuses, bins, CIs and valid replicate counts; six Pearson/Spearman columns |
| `reliability_family_summary.csv` | separate ICC2/ICC3 rows per method/band; spectral method-wide rows explicitly labeled; n_features/n_defined/n_undefined; mean, median, quartiles, IQR, threshold proportions |
| `reliability_pairwise.csv` | one row/feature/session pair; Pearson_r, Spearman_rho, status and same matched-unit counts |
| `reliability_matched_unit_manifest.csv` | subject/trial/window identity; present_s1/s2/s3, n_sessions_present, included_primary, missing_sessions, status; no labels/features |
| `subject_session_mean_reliability.csv` | optional separately named subject-mean analysis, same estimator schema |
| `coverage_report.json` | inherited ingestion provenance plus all matching/session/trial coverage and eligibility |
| `run_report.json` | estimator definitions, raw input declaration, bootstrap requested/executed counts and seed, feature lists, code/config/legacy notebook hashes and interpretation limits |

Family means/medians/IQR use all **defined** ICCs, including negatives. Threshold
proportions (`>=0.50`, `>=0.75`, `>=0.90`) have `n_defined` as denominator. Undefined
features stay visible in `n_features`/`n_undefined` and per-feature rows; they are not
silently dropped or counted as poor. Entirely undefined groups have blank summaries.
These are descriptions across features, not estimates from independent replicate
features. There is no significance-testing sweep or selection of reliable features.

Bins apply to point estimates only: poor `<0.50`, moderate `[0.50,0.75)`, good
`[0.75,0.90)`, excellent `>=0.90`; undefined remains separate. They are descriptive
conventions, not ground truth or universal fitness thresholds. Report estimates and
CIs alongside bins; see the interpretation discussion in
[Koo and Li (2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4913118/).

Figures are primary raw matched-unit results only:

- `icc2_distribution.png`: absolute-agreement ICC; negative values retained.
- `icc3_distribution.png`: consistency ICC, separately labeled.
- `icc_by_feature_family.png`: separate ICC2/ICC3 panels; median and feature IQR,
  explicitly not confidence intervals. Every PDC band remains separate.
- `pairwise_session_correlation.png`: separate Pearson/Spearman panels, labeled
  pairwise association/stability, not ICC.

Audit-only writes coverage, run report and matching manifest, but no ICC/correlation
tables, bootstrap CIs or figures. Synthetic tests use temporary output within 6.6
and remove it automatically.

## CLI

Safe validation from repository root:

```powershell
python -B phase_6_wearable_evaluation/6.6_session_reliability/code/test_session_reliability.py
python -B phase_6_wearable_evaluation/6.6_session_reliability/code/run_session_reliability.py --audit-only --bootstrap 0 --seed 42
```

Only when analysis is separately requested and inputs are eligible:

```powershell
python -B phase_6_wearable_evaluation/6.6_session_reliability/code/run_session_reliability.py --pilot --bootstrap 0 --datasets Spectral Connectivity Fusion
python -B phase_6_wearable_evaluation/6.6_session_reliability/code/run_session_reliability.py --require-all-subjects --bootstrap 1000 --seed 42 --datasets Spectral Connectivity Fusion --subject-session-means
```

Defaults: final coverage gate, all feature sets as a unique union, bootstrap=0,
seed=42, no optional subject means. Pilot/final flags are mutually exclusive.
Dependencies: numpy, pandas, scipy, matplotlib and the existing 6.2 ingestion
environment; no ICC package installation is required.

## Paper scope

Schema supports TRON feature-level/family tables, CIs, raw matched-unit figures and
auditable exclusions. No empirical final result exists until coverage is complete
and the real-data analysis is explicitly requested. This evidence is about offline
SEED measurement stability; it does not establish biomarker reproducibility,
clinical repeatability or longitudinal validation.

Do not describe the old Phase 5 ICC as test–retest reliability. Phase 6.6 is the first analysis that explicitly treats the three SEED sessions as repeated measurements. Preserve this distinction in README and later paper text.
