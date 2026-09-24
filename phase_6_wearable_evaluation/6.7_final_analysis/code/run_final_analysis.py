"""Phase 6.7: read-only integration of final Phase 6 evidence.

This module deliberately contains no imports from experiment runners.  It only reads
their serialized results, validates them, performs planned paired statistics, and
writes paper-facing tables and figures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[3]
STAGE = Path(__file__).resolve().parents[1]
PHASE_ROOT = STAGE.parent
DATASETS = ["Spectral", "Connectivity", "Fusion"]
PHASES = {
    "6.2": ("6.2_windowed_baseline", ["windowed_loso_per_subject.csv", "windowed_loso_summary.csv", "windowed_predictions.csv"]),
    "6.3": ("6.3_calibration_length", ["calibration_per_subject.csv", "calibration_summary.csv", "calibration_predictions.csv", "calibration_curve.csv"]),
    "6.4": ("6.4_cross_session", ["cross_session_per_subject.csv", "cross_session_summary.csv", "cross_session_predictions.csv"]),
    "6.5": ("6.5_streaming_normalization", ["streaming_per_subject.csv", "streaming_summary.csv", "streaming_predictions.csv", "streaming_cumulative_curve.csv"]),
    "6.6": ("6.6_session_reliability", ["reliability_per_feature.csv", "reliability_family_summary.csv", "reliability_pairwise.csv"]),
}
HISTORICAL = [
    ("Spectral", "Voting Soft", 0.7422, 0.0927),
    ("GC", "SVM Tuned", 0.6000, 0.1635),
    ("PDC Theta", "Voting Soft", 0.6015, 0.1106),
    ("Combined GC+PDC_ROI+Spectral", "Voting Soft", 0.7570, 0.0999),
    ("Combined", "SVM Tuned", 0.7526, 0.0852),
]


class IntegrationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise IntegrationError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _feature_fingerprint(report):
    value = report.get("feature_schema_hash") or report.get("feature_hash")
    if value:
        return value
    for key in ("feature_sets", "features", "feature_names"):
        if key in report:
            return canonical_hash(report[key])
    return None


def _parameter_fingerprint(report):
    return (report.get("parameter_hash") or report.get("parameter_sha256") or
            (canonical_hash(report["parameters"]) if "parameters" in report else None))


def _source_fingerprint(report, coverage):
    selected = coverage.get("selected_runs", [])
    if selected:
        compact = [{k: row.get(k) for k in ("subject_num", "session", "trial", "report_sha256", "csv_sha256")}
                   for row in selected]
        return canonical_hash(compact)
    for key in ("source_hash", "input_hash", "input_provenance_hash"):
        if key in report:
            return report[key]
    return None


def full_coverage(coverage) -> tuple[bool, str]:
    complete = coverage.get("complete_subjects")
    if complete is not None:
        ok = len(set(map(int, complete))) == 15
        return ok, f"complete_subjects={len(set(complete))}/15"
    rows = coverage.get("coverage") or coverage.get("per_subject")
    if isinstance(rows, list) and rows:
        flags = [bool(r.get("complete", r.get("full_coverage", False))) for r in rows]
        return len(rows) == 15 and all(flags), f"complete rows={sum(flags)}/{len(rows)}"
    for key in ("full_coverage", "coverage_complete", "final_eligible"):
        if key in coverage:
            return bool(coverage[key]), f"{key}={coverage[key]}"
    return False, "coverage report has no recognized full-coverage proof"


def candidate_runs(phase_root: Path, phase: str):
    folder, required_csv = PHASES[phase]
    base = phase_root / folder / "output" / "final"
    candidates = []
    for report_path in sorted(base.glob("*/run_report.json")):
        run = report_path.parent
        item = {"path": str(run.resolve()), "report_path": str(report_path.resolve()), "valid": False, "reasons": []}
        try:
            report = json_read(report_path)
            coverage_path = run / "coverage_report.json"
            if report.get("status") != "passed": item["reasons"].append("status is not passed")
            if report.get("mode") != "final": item["reasons"].append("mode is not final")
            if report.get("audit_only") is True: item["reasons"].append("audit_only run")
            missing = [name for name in required_csv + ["coverage_report.json"] if not (run / name).is_file()]
            if missing: item["reasons"].append(f"missing files: {missing}")
            coverage = json_read(coverage_path) if coverage_path.is_file() else {}
            covered, coverage_note = full_coverage(coverage)
            if not covered: item["reasons"].append(f"incomplete coverage: {coverage_note}")
            item.update(report=report, coverage=coverage, coverage_note=coverage_note,
                        timestamp=report.get("timestamp") or run.name,
                        feature_hash=_feature_fingerprint(report), parameter_hash=_parameter_fingerprint(report),
                        source_hash=_source_fingerprint(report, coverage))
            item["compatibility_fingerprint"] = canonical_hash({
                "feature": item["feature_hash"], "parameter": item["parameter_hash"],
                "source": item["source_hash"], "protocol": report.get("protocol"),
                "normalization": report.get("normalization"), "datasets": report.get("datasets")})
            item["valid"] = not item["reasons"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            item["reasons"].append(f"unreadable/malformed: {exc}")
        candidates.append(item)
    return candidates


def select_latest_run(phase_root: Path, phase: str):
    candidates = candidate_runs(phase_root, phase)
    valid = [c for c in candidates if c["valid"]]
    if not valid:
        return None, candidates
    fingerprints = {c["compatibility_fingerprint"] for c in valid}
    if len(fingerprints) > 1:
        detail = [{"path": c["path"], "feature_hash": c["feature_hash"],
                   "parameter_hash": c["parameter_hash"], "source_hash": c["source_hash"]} for c in valid]
        raise IntegrationError(f"{phase}: ambiguous incompatible passed final runs: {detail}")
    return sorted(valid, key=lambda c: (str(c["timestamp"]), c["path"]))[-1], candidates


def provenance_entry(phase, selected):
    run = Path(selected["path"])
    files = PHASES[phase][1] + ["run_report.json", "coverage_report.json"]
    report = selected["report"]
    return {
        "selected_run_path": str(run), "run_timestamp": selected["timestamp"],
        "code_hash": report.get("code_hash") or report.get("script_sha256"),
        "config_hash": report.get("config_hash") or report.get("config_sha256"),
        "feature_schema_hash": selected["feature_hash"], "parameter_hash": selected["parameter_hash"],
        "source_input_hash": selected["source_hash"],
        "source_output_hashes": {name: sha256(run / name) for name in files},
    }


def read_csv(run: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(run / name)
    require(len(frame) > 0, f"{name}: empty table")
    require(frame.columns.is_unique, f"{name}: duplicate columns")
    return frame


def require_columns(frame, columns, name):
    missing = sorted(set(columns) - set(frame.columns))
    require(not missing, f"{name}: missing columns {missing}")


def finite(frame, columns, name, allow_nan=()):
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if column in allow_nan:
            require(not np.isinf(values).any(), f"{name}: infinite {column}")
        else:
            require(np.isfinite(values).all(), f"{name}: non-finite {column}")


def unique_rows(frame, keys, name):
    require_columns(frame, keys, name)
    duplicate = frame.duplicated(keys, keep=False)
    require(not duplicate.any(), f"{name}: duplicate rows for {keys}: {frame.loc[duplicate, keys].head().to_dict('records')}")


def validate_datasets(frame, name):
    require_columns(frame, ["dataset"], name)
    found = set(frame.dataset.astype(str))
    require(found == set(DATASETS), f"{name}: datasets {sorted(found)} != {DATASETS}")


def validate_inputs(selected, tolerance):
    tables = {}
    for phase, item in selected.items():
        run = Path(item["path"])
        tables[phase] = {name: read_csv(run, name) for name in PHASES[phase][1]}

    p62 = tables["6.2"]["windowed_loso_per_subject.csv"]
    require_columns(p62, ["dataset", "subject_num", "accuracy", "macro_f1", "weighted_f1"], "6.2 per-subject")
    validate_datasets(p62, "6.2 per-subject"); unique_rows(p62, ["dataset", "subject_num"], "6.2 per-subject")
    finite(p62, ["accuracy", "macro_f1", "weighted_f1"], "6.2 per-subject")
    require(p62.subject_num.nunique() == 15 and (p62.groupby("dataset").subject_num.nunique() == 15).all(), "6.2: zero-cal coverage must be exactly 15 subjects per dataset")

    p63 = tables["6.3"]["calibration_per_subject.csv"]
    require_columns(p63, ["dataset", "subject_num", "condition", "accuracy", "macro_f1", "matched_gain_pp", "matched_gap_to_full_pp", "matched_recovery_pct"], "6.3 per-subject")
    validate_datasets(p63, "6.3 per-subject"); unique_rows(p63, ["dataset", "subject_num", "condition"], "6.3 per-subject")
    finite(p63, ["accuracy", "macro_f1", "matched_gain_pp", "matched_gap_to_full_pp", "matched_recovery_pct"], "6.3 per-subject", allow_nan=("matched_recovery_pct",))
    require(set(["0", "1", "2", "5", "10", "full"]) <= set(p63.condition.astype(str)), "6.3: required calibration conditions missing")

    p64 = tables["6.4"]["cross_session_per_subject.csv"]
    require_columns(p64, ["dataset", "subject_num", "condition", "accuracy", "macro_f1"], "6.4 per-subject")
    validate_datasets(p64, "6.4 per-subject"); unique_rows(p64, ["dataset", "subject_num", "condition"], "6.4 per-subject")
    finite(p64, ["accuracy", "macro_f1"], "6.4 per-subject")
    require(set(p64.condition) == {"zero_cal", "session1_cal"}, "6.4: conditions must be zero_cal/session1_cal")

    p65 = tables["6.5"]["streaming_per_subject.csv"]
    require_columns(p65, ["dataset", "subject_num", "condition", "accuracy", "macro_f1", "gain_vs_zero_pp", "gain_vs_fixed_pp"], "6.5 per-subject")
    validate_datasets(p65, "6.5 per-subject"); unique_rows(p65, ["dataset", "subject_num", "condition"], "6.5 per-subject")
    finite(p65, ["accuracy", "macro_f1", "gain_vs_zero_pp", "gain_vs_fixed_pp"], "6.5 per-subject")
    require(set(p65.condition) == {"zero_cal", "fixed_session1", "streaming_running", "streaming_ema"}, "6.5: required conditions missing")

    p66 = tables["6.6"]["reliability_per_feature.csv"]
    require_columns(p66, ["feature_name", "feature_family", "method", "band", "ICC2_1", "ICC3_1", "ICC2_1_status", "ICC3_1_status"], "6.6 per-feature")
    unique_rows(p66, ["feature_name"], "6.6 per-feature")
    finite(p66, ["ICC2_1", "ICC3_1"], "6.6 per-feature", allow_nan=("ICC2_1", "ICC3_1"))
    for metric in ("ICC2_1", "ICC3_1"):
        ok = p66[f"{metric}_status"].eq("ok")
        require(np.isfinite(pd.to_numeric(p66.loc[ok, metric], errors="coerce")).all(), f"6.6: finite {metric} required when status=ok")
        require(p66.loc[~ok, metric].isna().all(), f"6.6: undefined {metric} must be blank/NaN")

    # Same-protocol identities are exact subject-level checks, not aggregate approximations.
    z62 = p62.set_index(["dataset", "subject_num"])
    z63 = p63[p63.condition.astype(str) == "0"].set_index(["dataset", "subject_num"])
    require(set(z62.index) == set(z63.index), "6.3 condition 0 subject keys differ from 6.2")
    for metric in ("accuracy", "macro_f1", "weighted_f1"):
        mismatch = np.abs(z62[metric] - z63[metric]) > tolerance
        require(not mismatch.any(), f"6.3 condition 0 vs 6.2 {metric} mismatch: {list(z62.index[mismatch])[:5]}")

    map64 = {"zero_cal": "zero_cal", "session1_cal": "fixed_session1"}
    for c64, c65 in map64.items():
        a = p64[p64.condition == c64].set_index(["dataset", "subject_num"])
        b = p65[p65.condition == c65].set_index(["dataset", "subject_num"])
        require(set(a.index) == set(b.index), f"6.4 {c64}/6.5 {c65}: subject keys differ")
        for metric in ("accuracy", "macro_f1"):
            mismatch = np.abs(a[metric] - b[metric]) > tolerance
            require(not mismatch.any(), f"6.5 {c65} mismatch with 6.4 {c64} for {metric}: {list(a.index[mismatch])[:5]}")
    return tables


def mean_sd(series):
    s = pd.to_numeric(series, errors="raise")
    return float(s.mean()), float(s.std(ddof=1))


def build_tables(tables):
    p62 = tables["6.2"]["windowed_loso_per_subject.csv"]
    rows = []
    for dataset in DATASETS:
        b = p62[p62.dataset == dataset]
        am, ass = mean_sd(b.accuracy); fm, fs = mean_sd(b.macro_f1); wm, ws = mean_sd(b.weighted_f1)
        rows.append({"Dataset": dataset, "Accuracy mean": am, "Accuracy SD": ass,
                     "Accuracy mean ± SD": f"{am:.3f} ± {ass:.3f}", "Macro F1 mean": fm, "Macro F1 SD": fs,
                     "Macro F1 mean ± SD": f"{fm:.3f} ± {fs:.3f}", "Weighted F1 mean": wm, "Weighted F1 SD": ws,
                     "Weighted F1 mean ± SD": f"{wm:.3f} ± {ws:.3f}", "n_subjects": int(b.subject_num.nunique())})
    baseline = pd.DataFrame(rows)

    p63 = tables["6.3"]["calibration_per_subject.csv"].copy(); p63["condition"] = p63.condition.astype(str)
    rows = []
    for dataset in DATASETS:
        for condition in ["0", "1", "2", "5", "10", "full"]:
            b = p63[(p63.dataset == dataset) & (p63.condition == condition)]
            am, ass = mean_sd(b.accuracy); fm, fs = mean_sd(b.macro_f1)
            rows.append({"Dataset": dataset, "Calibration": condition, "Accuracy mean": am, "Accuracy SD": ass,
                         "Accuracy mean ± SD": f"{am:.3f} ± {ass:.3f}", "Macro F1 mean": fm, "Macro F1 SD": fs,
                         "Macro F1 mean ± SD": f"{fm:.3f} ± {fs:.3f}",
                         "Matched gain vs zero": float(b.matched_gain_pp.mean()),
                         "Gap to full reference": float(b.matched_gap_to_full_pp.mean()),
                         "Recovery %": float(b.matched_recovery_pct.mean()) if b.matched_recovery_pct.notna().any() else np.nan,
                         "n_subjects": int(b.subject_num.nunique()), "oracle_reference": condition == "full"})
    calibration = pd.DataFrame(rows)

    p64 = tables["6.4"]["cross_session_per_subject.csv"]
    rows = []
    for dataset in DATASETS:
        pivot = p64[p64.dataset == dataset].pivot(index="subject_num", columns="condition", values=["accuracy", "macro_f1"])
        gain = 100 * (pivot[("accuracy", "session1_cal")] - pivot[("accuracy", "zero_cal")])
        rows.append({"Dataset": dataset, "zero_cal accuracy": float(pivot[("accuracy", "zero_cal")].mean()),
                     "session1_cal accuracy": float(pivot[("accuracy", "session1_cal")].mean()),
                     "paired gain_pp": float(gain.mean()), "zero_cal macro F1": float(pivot[("macro_f1", "zero_cal")].mean()),
                     "session1_cal macro F1": float(pivot[("macro_f1", "session1_cal")].mean()), "n_subjects": len(pivot)})
    cross = pd.DataFrame(rows)

    p65 = tables["6.5"]["streaming_per_subject.csv"]
    rows = []
    for dataset in DATASETS:
        for condition in ["zero_cal", "fixed_session1", "streaming_running", "streaming_ema"]:
            b = p65[(p65.dataset == dataset) & (p65.condition == condition)]
            am, ass = mean_sd(b.accuracy); fm, fs = mean_sd(b.macro_f1)
            rows.append({"Dataset": dataset, "Condition": condition, "Accuracy mean": am, "Accuracy SD": ass,
                         "Accuracy mean ± SD": f"{am:.3f} ± {ass:.3f}", "Macro F1 mean": fm, "Macro F1 SD": fs,
                         "Macro F1 mean ± SD": f"{fm:.3f} ± {fs:.3f}", "gain_vs_zero_pp": float(b.gain_vs_zero_pp.mean()),
                         "gain_vs_fixed_pp": float(b.gain_vs_fixed_pp.mean()), "n_subjects": int(b.subject_num.nunique())})
    streaming = pd.DataFrame(rows)

    p66 = tables["6.6"]["reliability_per_feature.csv"]
    rows = []
    for keys, b in p66.groupby(["feature_family", "method", "band"], sort=False, dropna=False):
        i2 = pd.to_numeric(b.ICC2_1, errors="coerce").dropna(); i3 = pd.to_numeric(b.ICC3_1, errors="coerce").dropna()
        rows.append({"feature_family": keys[0], "method": keys[1], "band": keys[2], "n_features": len(b),
                     "n_defined_ICC2": len(i2), "median ICC2": i2.median(), "IQR ICC2": i2.quantile(.75)-i2.quantile(.25),
                     "n_defined_ICC3": len(i3), "median ICC3": i3.median(), "IQR ICC3": i3.quantile(.75)-i3.quantile(.25),
                     "proportion ICC2 >= 0.50": (i2 >= .5).mean() if len(i2) else np.nan,
                     "proportion ICC3 >= 0.50": (i3 >= .5).mean() if len(i3) else np.nan})
    reliability = pd.DataFrame(rows)
    return baseline, calibration, cross, streaming, reliability


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    require(np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all(), "Invalid p-values for Holm correction")
    order = np.argsort(p); adjusted = np.empty(len(p)); running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_comparison(frame, condition_a, condition_b, comparison, dataset):
    block = frame[frame.dataset == dataset]
    a = block[block.condition.astype(str) == condition_a].set_index("subject_num").accuracy
    b = block[block.condition.astype(str) == condition_b].set_index("subject_num").accuracy
    subjects = sorted(set(a.index) & set(b.index))
    require(subjects and set(a.index) == set(b.index), f"{comparison}/{dataset}: paired subjects differ")
    diff = b.loc[subjects].to_numpy(float) - a.loc[subjects].to_numpy(float)
    if np.all(np.abs(diff) == 0):
        statistic, pvalue = 0.0, 1.0
    else:
        result = wilcoxon(diff, alternative="two-sided", zero_method="wilcox", method="auto")
        statistic, pvalue = float(result.statistic), float(result.pvalue)
    return {"comparison": comparison, "dataset": dataset, "condition_a": condition_a, "condition_b": condition_b,
            "n_pairs": len(subjects), "wilcoxon_statistic": statistic, "raw_p": pvalue,
            "mean_paired_difference_pp": float(100 * diff.mean()), "median_paired_difference_pp": float(100 * np.median(diff)),
            "paired_subject_ids": ";".join(map(str, subjects))}


def matched_gain_comparison(frame, condition, dataset):
    """Test Phase 6.3's subject-level same-evaluation-row gain against zero."""
    block = frame[(frame.dataset == dataset) & (frame.condition.astype(str) == condition)].copy()
    require_columns(block, ["subject_num", "matched_gain_pp"], "6.3 matched comparison")
    unique_rows(block, ["subject_num"], "6.3 matched comparison")
    subjects = sorted(block.subject_num.astype(int).tolist())
    require(len(subjects) == 15, f"6.3 {condition}/{dataset}: expected 15 paired subjects")
    diff = block.set_index("subject_num").loc[subjects, "matched_gain_pp"].to_numpy(float) / 100.0
    require(np.isfinite(diff).all(), f"6.3 {condition}/{dataset}: non-finite matched gains")
    if np.all(np.abs(diff) == 0):
        statistic, pvalue = 0.0, 1.0
    else:
        result = wilcoxon(diff, alternative="two-sided", zero_method="wilcox", method="auto")
        statistic, pvalue = float(result.statistic), float(result.pvalue)
    return {"comparison": f"6.3 {condition} min vs zero on exact same evaluation rows", "dataset": dataset,
            "condition_a": "accuracy_0_same_rows", "condition_b": condition, "n_pairs": len(subjects),
            "wilcoxon_statistic": statistic, "raw_p": pvalue,
            "mean_paired_difference_pp": float(100 * diff.mean()),
            "median_paired_difference_pp": float(100 * np.median(diff)),
            "paired_subject_ids": ";".join(map(str, subjects))}


def planned_tests(tables):
    rows = []
    p63 = tables["6.3"]["calibration_per_subject.csv"].copy(); p63.condition = p63.condition.astype(str)
    for dataset in DATASETS:
        for condition in ["1", "2", "5", "10"]:
            rows.append(matched_gain_comparison(p63, condition, dataset))
    p64 = tables["6.4"]["cross_session_per_subject.csv"]
    p65 = tables["6.5"]["streaming_per_subject.csv"]
    for dataset in DATASETS:
        rows.append(paired_comparison(p64, "zero_cal", "session1_cal", "6.4 session1_cal vs zero_cal", dataset))
        rows.append(paired_comparison(p65, "fixed_session1", "streaming_running", "6.5 streaming_running vs fixed_session1", dataset))
        rows.append(paired_comparison(p65, "fixed_session1", "streaming_ema", "6.5 streaming_ema vs fixed_session1", dataset))
    result = pd.DataFrame(rows)
    # One predeclared family across the limited number of planned comparisons.
    result["holm_corrected_p"] = holm_adjust(result.raw_p.to_numpy())
    result["significant_holm_0.05"] = result["holm_corrected_p"] < .05
    return result


def historical_table():
    return pd.DataFrame([{"dataset": d, "model": m, "accuracy_mean": mean, "accuracy_sd": sd,
                          "accuracy_mean_percent": 100*mean, "accuracy_sd_percent": 100*sd,
                          "analysis_level": "trial", "normalization": "full_subject_unlabeled_reference",
                          "protocol": "historical_phase5_loso"} for d, m, mean, sd in HISTORICAL])


def save_figures(tables_out, tables, figures):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures.mkdir(parents=True, exist_ok=True)
    baseline, calibration, cross, streaming, reliability = tables_out

    fig, ax = plt.subplots(figsize=(8, 5)); x = np.arange(3); width = .35
    ax.bar(x-width/2, baseline["Accuracy mean"], width, yerr=baseline["Accuracy SD"], label="Accuracy", capsize=3)
    ax.bar(x+width/2, baseline["Macro F1 mean"], width, yerr=baseline["Macro F1 SD"], label="Macro F1", capsize=3)
    ax.set(xticks=x, xticklabels=DATASETS, ylim=(0, 1), ylabel="Score", title="Unseen-subject zero-calibration performance"); ax.legend(); fig.tight_layout(); fig.savefig(figures/"fig_main_baseline.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset in DATASETS:
        b = calibration[(calibration.Dataset == dataset) & calibration.Calibration.isin(["0","1","2","5","10"])]
        ax.errorbar(pd.to_numeric(b.Calibration), b["Accuracy mean"], yerr=b["Accuracy SD"], marker="o", capsize=3, label=dataset)
        oracle = calibration[(calibration.Dataset == dataset) & (calibration.Calibration == "full")]["Accuracy mean"].iloc[0]
        ax.axhline(oracle, linestyle="--", alpha=.25)
    ax.set(xlabel="Unlabeled calibration (minutes)", ylabel="Accuracy", ylim=(0,1), title="Calibration-length evaluation"); ax.legend(); fig.tight_layout(); fig.savefig(figures/"fig_calibration_curve.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,5)); p64=tables["6.4"]["cross_session_per_subject.csv"]
    for i,d in enumerate(DATASETS):
        b=p64[p64.dataset==d].pivot(index="subject_num",columns="condition",values="accuracy")
        for _,row in b.iterrows(): ax.plot([i-.17,i+.17],[row.zero_cal,row.session1_cal],color="0.75",linewidth=.7)
        ax.scatter(np.full(len(b),i-.17),b.zero_cal,label="zero_cal" if i==0 else None); ax.scatter(np.full(len(b),i+.17),b.session1_cal,label="session1_cal" if i==0 else None)
    ax.set(xticks=range(3),xticklabels=DATASETS,ylabel="Accuracy",ylim=(0,1),title="Held-out Session-3 paired performance"); ax.legend(); fig.tight_layout(); fig.savefig(figures/"fig_cross_session.png",dpi=300); plt.close(fig)

    fig, ax=plt.subplots(figsize=(10,5)); conditions=["zero_cal","fixed_session1","streaming_running","streaming_ema"]
    for i,c in enumerate(conditions):
        b=streaming[streaming.Condition==c].set_index("Dataset").loc[DATASETS]
        ax.bar(np.arange(3)+(i-1.5)*.2,b["Accuracy mean"],.2,yerr=b["Accuracy SD"],capsize=2,label=c)
    ax.set(xticks=range(3),xticklabels=DATASETS,ylabel="Accuracy",ylim=(0,1),title="Causal streaming-normalization conditions"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(figures/"fig_streaming_adaptation.png",dpi=300); plt.close(fig)

    labels=(reliability.method.astype(str)+"\n"+reliability.band.astype(str)).tolist(); x=np.arange(len(labels)); fig,ax=plt.subplots(figsize=(max(10,len(labels)*.65),5))
    ax.errorbar(x-.12,reliability["median ICC2"],yerr=reliability["IQR ICC2"]/2,fmt="o",capsize=3,label="ICC(2,1)")
    ax.errorbar(x+.12,reliability["median ICC3"],yerr=reliability["IQR ICC3"]/2,fmt="s",capsize=3,label="ICC(3,1)")
    vals=pd.concat([reliability["median ICC2"]-reliability["IQR ICC2"],reliability["median ICC3"]-reliability["IQR ICC3"]]); lower=min(-.1,float(vals.min(skipna=True))-.05)
    ax.set(xticks=x,xticklabels=labels,ylabel="Median ICC (feature IQR shown)",ylim=(lower,1),title="Raw feature session reliability")
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels(): label.set_ha("right")
    ax.axhline(.5,color="0.7",linestyle="--"); ax.legend(); fig.tight_layout(); fig.savefig(figures/"fig_reliability.png",dpi=300); plt.close(fig)


def paper_summary(tables_out, tests):
    baseline, calibration, cross, streaming, reliability = tables_out
    def best_text(frame, value, label):
        row=frame.loc[frame[value].idxmax()]; return f"{row[label]} ({row[value]:.3f})"
    fusion=baseline.set_index("Dataset").loc["Fusion","Accuracy mean"]; spectral=baseline.set_index("Dataset").loc["Spectral","Accuracy mean"]
    rq1 = (f"Across 15 held-out subjects, the highest mean zero-calibration accuracy was {best_text(baseline,'Accuracy mean','Dataset')}. " +
           ("Fusion showed a descriptive complementary gain over Spectral. " if fusion > spectral else "Connectivity fusion did not improve mean window-level generalization over Spectral. ") +
           "This is descriptive and is not compared inferentially with historical Phase 5.")
    limited=calibration[calibration.Calibration.isin(["1","2","5","10"])]
    gains=limited.groupby("Dataset")["Matched gain vs zero"].max()
    rq2="Matched-row gains were heterogeneous across datasets; the largest matched gain for each dataset was " + ", ".join(f"{k}: {v:.2f} pp" for k,v in gains.items()) + ". Full is a transductive oracle/reference, not a deployment condition."
    rq3="Session-1 calibration paired gains on held-out Session 3 were " + ", ".join(f"{r.Dataset}: {r['paired gain_pp']:.2f} pp" for _,r in cross.iterrows()) + "."
    run=streaming[streaming.Condition=="streaming_running"].set_index("Dataset")["gain_vs_fixed_pp"]; ema=streaming[streaming.Condition=="streaming_ema"].set_index("Dataset")["gain_vs_fixed_pp"]
    rq4="Relative to fixed Session-1 normalization, running/EMA gains (pp) were " + ", ".join(f"{d}: {run[d]:.2f}/{ema[d]:.2f}" for d in DATASETS) + ". These offline prequential results evaluate causal streaming normalization, not wall-clock real-time performance."
    med2=reliability["median ICC2"].median(); med3=reliability["median ICC3"].median()
    rq5=f"Across all reported method/band groups without cherry-picking, the median of group median ICC(2,1) was {med2:.3f} and ICC(3,1) was {med3:.3f}. Negative estimates and undefined features remain represented in source/final tables."
    return f"""# Paper-ready Phase 6 results summary

## Dataset/protocol

Phase 6 is an offline deployment-oriented evaluation using 60-s window-level features: subject-independent zero/limited calibration, held-out cross-session testing, causal streaming normalization, and raw-feature session reliability. Historical Phase 5 is kept in a separate table because it used trial-level features and full held-out-subject unlabeled normalization under historical LOSO; the two protocols are not apples-to-apples.

## RQ1

{rq1}

## RQ2

{rq2}

## RQ3

{rq3}

## RQ4

{rq4}

## RQ5

{rq5}

## Limitations

These findings support an offline deployment-oriented evaluation toward wearable monitoring, not wearable deployment readiness, clinical validation, or real-time validation. Historical hyperparameters were developed on the same cohort; Phase 6 is not independent nested model selection. Limited-calibration conditions exclude their calibration rows, full calibration is transductive, SEED repeats stimuli across sessions, and reliability reflects the specified complete-case repeated-session design. Planned Wilcoxon tests use matched subject IDs with Holm correction; effect magnitudes and null/negative findings are retained regardless of p-value.
"""


def audit(phase_root: Path):
    report={"status":"audit_only","all_prerequisites_available":False,"phases":{}}
    selected={}; errors=[]
    for phase in PHASES:
        try:
            choice,candidates=select_latest_run(phase_root,phase)
            report["phases"][phase]={"available":choice is not None,"selected_run":choice["path"] if choice else None,
                                      "candidates":[{"path":c["path"],"valid":c["valid"],"reasons":c["reasons"]} for c in candidates]}
            if choice: selected[phase]=choice
        except IntegrationError as exc:
            errors.append(str(exc)); report["phases"][phase]={"available":False,"selected_run":None,"error":str(exc)}
    report["all_prerequisites_available"]=len(selected)==len(PHASES) and not errors
    report["missing_phases"]=[p for p in PHASES if p not in selected]
    report["errors"]=errors
    return report,selected


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only",action="store_true")
    parser.add_argument("--require-complete",action="store_true")
    parser.add_argument("--tolerance",type=float,default=1e-10)
    parser.add_argument("--phase-root",type=Path,default=PHASE_ROOT,help=argparse.SUPPRESS)
    parser.add_argument("--output-root",type=Path,default=STAGE/"output",help=argparse.SUPPRESS)
    parser.add_argument("--figures-root",type=Path,default=STAGE/"figures",help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    require(args.tolerance>=0 and math.isfinite(args.tolerance),"tolerance must be finite and non-negative")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    mode="audit" if args.audit_only else "final"; out=args.output_root/mode/stamp; out.mkdir(parents=True,exist_ok=False)
    prereq,selected=audit(args.phase_root); json_write(out/"prerequisite_report.json",prereq)
    provenance={p:provenance_entry(p,s) for p,s in selected.items()}; json_write(out/"upstream_provenance.json",provenance)
    if args.audit_only:
        json_write(out/"run_report.json",{"status":"audit_only","mode":"audit","timestamp":stamp,"final_claims_generated":False,"selected_phases":sorted(selected),"missing_phases":prereq["missing_phases"],"script_sha256":sha256(Path(__file__))})
        print(f"Audit complete; selected={sorted(selected)}, missing={prereq['missing_phases']}; {out}")
        return 0
    try:
        require(prereq["all_prerequisites_available"],f"Final integration requires passed final runs for 6.2-6.6; missing={prereq['missing_phases']}, errors={prereq['errors']}")
        tables=validate_inputs(selected,args.tolerance)
        final_tables=build_tables(tables); names=["final_baseline_table.csv","final_calibration_table.csv","final_cross_session_table.csv","final_streaming_table.csv","final_reliability_table.csv"]
        for frame,name in zip(final_tables,names): frame.to_csv(out/name,index=False)
        tests=planned_tests(tables); tests.to_csv(out/"planned_pairwise_tests.csv",index=False)
        historical_table().to_csv(out/"historical_phase5_reference.csv",index=False)
        figures=args.figures_root/"final"/stamp; save_figures(final_tables,tables,figures)
        (out/"paper_results_summary.md").write_text(paper_summary(final_tables,tests),encoding="utf-8")
        prereq["status"]="passed"; prereq["consistency_checks_passed"]=True; json_write(out/"prerequisite_report.json",prereq)
        json_write(out/"run_report.json",{"status":"passed","mode":"final","timestamp":stamp,"tolerance":args.tolerance,"upstream_paths":{p:s["path"] for p,s in selected.items()},"figures_path":str(figures.resolve()),"no_recomputation":True,"script_sha256":sha256(Path(__file__))})
        print(f"Final paper integration passed: {out}"); return 0
    except (IntegrationError,OSError,ValueError,KeyError) as exc:
        prereq["status"]="rejected"; prereq["rejection_reason"]=str(exc); json_write(out/"prerequisite_report.json",prereq)
        json_write(out/"run_report.json",{"status":"rejected","mode":"final","timestamp":stamp,"reason":str(exc),"final_claims_generated":False})
        print(f"Rejected: {exc}",file=sys.stderr); return 2


if __name__=="__main__":
    raise SystemExit(main())
