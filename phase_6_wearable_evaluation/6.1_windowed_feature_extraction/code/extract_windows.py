"""60-second within-trial extraction; defaults to S01/session 1/trial 1."""
import argparse
import ast
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as cfg


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def notebook_definitions(relative, names, namespace):
    """Reuse selected definitions only; never execute notebook I/O or experiments."""
    notebook = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    found = set()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            name = node.name if isinstance(node, ast.FunctionDef) else None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                name = getattr(node.targets[0], "id", None)
            if name in names:
                exec(compile(ast.Module(body=[node], type_ignores=[]), relative, "exec"), namespace)
                found.add(name)
    if found != set(names):
        raise ValueError(f"Missing notebook definitions: {set(names) - found}")
    return namespace


SPECTRAL = "03_spectral_features/code/03_spectral_features.ipynb"
ROI = "phase_3_feature_engineering/3.4_roi_aggregation/code/3.4_roi_aggregation.ipynb"
GC = "01_granger_causality/code/gc_analyzer.py"
PDC = "02_pdc/code/pdc_analyzer.py"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gc_audit(raw, pvalues, thresholded, alpha):
    """Descriptive audit only; it does not apply or change thresholding."""
    offdiag = ~np.eye(raw.shape[0], dtype=bool)
    raw_values = raw[offdiag]
    p = pvalues[offdiag]
    threshold_values = thresholded[offdiag]
    return {
        "gc_raw_nonzero_density": float(np.count_nonzero(raw_values) / len(raw_values)),
        "gc_thresholded_density": float(np.count_nonzero(threshold_values) / len(threshold_values)),
        "gc_pvalue_min": float(np.min(p)),
        "gc_pvalue_q01": float(np.quantile(p, .01)),
        "gc_pvalue_q05": float(np.quantile(p, .05)),
        "gc_pvalue_median": float(np.median(p)),
        "gc_pvalue_q95": float(np.quantile(p, .95)),
        "gc_pvalue_max": float(np.max(p)),
        "gc_uncorrected_p_lt_alpha": int(np.count_nonzero(p < alpha)),
        # For the configured FDR method, nonzero thresholded edges are exactly
        # the rejected edges. Other methods retain this count under a neutral name.
        "gc_fdr_rejected_edges": int(np.count_nonzero(threshold_values)),
        "gc_possible_edges": int(len(p)),
    }


def window_bounds(n_samples, fs=100, seconds=60):
    size = int(fs * seconds)
    if size <= 0 or size != fs * seconds or n_samples < 0:
        raise ValueError("Invalid window length or sample count")
    return [(start, start + size) for start in range(0, n_samples - size + 1, size)]


def run(subject=1, session_index=1, trial=1):
    run_started = time.perf_counter()
    if subject not in cfg.SUBJECT_SESSIONS or not 1 <= session_index <= 3 or not 1 <= trial <= 15:
        raise ValueError("Expected subject 1..15, session-index 1..3, trial 1..15")
    session = cfg.SUBJECT_SESSIONS[subject][session_index - 1]
    source = Path(cfg.PREPROCESS_DIR) / f"subject_{subject:02d}" / f"session_{session}" / f"trial_{trial:02d}.npy"
    load_started = time.perf_counter()
    eeg = np.load(source, allow_pickle=False)
    if eeg.ndim != 2 or eeg.shape[0] != len(cfg.CHANNEL_NAMES) or not np.isfinite(eeg).all():
        raise ValueError("Expected finite EEG with shape (62, samples)")
    bounds = window_bounds(eeg.shape[1], cfg.TARGET_FS)
    if not bounds:
        raise ValueError("Trial has no full 60-second window")
    gc = load_module("window_gc", GC)
    pdc = load_module("window_pdc", PDC)
    spectral = notebook_definitions(SPECTRAL, ["extract_trial_features"], {"np": np, "welch": welch})
    roi = notebook_definitions(ROI, ["ROI_GROUPS", "aggregate_roi_matrix", "extract_roi_features"], {"np": np})
    roi_names = list(roi["ROI_GROUPS"])
    roi_indices = {name: [cfg.CHANNEL_NAMES.index(c) for c in channels]
                   for name, channels in roi["ROI_GROUPS"].items()}
    load_preprocess_seconds = time.perf_counter() - load_started
    out = STAGE / "output" / f"subject_{subject:02d}" / f"session_{session}" / f"trial_{trial:02d}"
    out.mkdir(parents=True, exist_ok=True)
    # Unique run directory prevents mixing partial/new results with a previous run.
    from datetime import datetime, timezone
    out = out / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    out.mkdir()
    provenance = {p: sha256(ROOT / p)
                  for p in ["config.py", GC, PDC, SPECTRAL, ROI]}
    provenance[str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/")] = sha256(__file__)
    metadata = {
        "status": "running", "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "pipeline_sha256": provenance, "fs": cfg.TARGET_FS,
        "window_seconds": 60, "overlap_samples": 0, "tail_policy": "drop",
        "input_shape": list(eeg.shape), "n_windows": len(bounds),
        "discarded_samples": eeg.shape[1] - bounds[-1][1],
        "channels": cfg.CHANNEL_NAMES, "gc_max_lag": cfg.GC_MAX_LAG,
        "gc_threshold": cfg.GC_THRESHOLD_METHOD, "gc_alpha": cfg.GC_ALPHA,
        "pdc_max_order": cfg.PDC_MAX_ORDER, "pdc_min_order": cfg.PDC_MIN_ORDER,
        "bands": cfg.PDC_FREQUENCY_BANDS,
        "matrix_orientation": "GC: source,target; PDC raw: target,source; ROI features: source,target",
        "normalization": "GC internal per-window preprocessing only; no subject/session feature normalization",
        "scope": "Offline windows from already-preprocessed EEG; not causal streaming preprocessing",
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
    }
    report = out / "sanity_report.json"
    report.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rows, manifest = [], []
    for number, (start, stop) in enumerate(bounds, 1):
        t0 = time.perf_counter()
        window = eeg[:, start:stop]
        if np.any(window.std(axis=1) < 1e-10):
            raise ValueError(f"Constant channel in window {number}")
        key = {"subject_id": f"S{subject:02d}", "subject_num": subject,
               "session": int(session), "session_index": session_index, "trial": trial,
               "class": cfg.LABEL_MAP[cfg.TRIAL_LABELS[trial - 1]],
               "class_label": cfg.EMOTION_MAP[cfg.TRIAL_LABELS[trial - 1]],
               "window_id": number, "start_sample": start, "stop_sample": stop,
               "start_seconds": start / cfg.TARGET_FS, "stop_seconds": stop / cfg.TARGET_FS}
        print(f"Window {number}/{len(bounds)} [{start}:{stop}]: GC", flush=True)
        part_started = time.perf_counter()
        raw, pv, thresholded, density = gc.analyze_trial(window)
        gc_seconds = time.perf_counter() - part_started
        print(f"Window {number}: PDC", flush=True)
        part_started = time.perf_counter()
        band_pdc, order = pdc.analyze_trial(window)
        pdc_seconds = time.perf_counter() - part_started
        part_started = time.perf_counter()
        features = spectral["extract_trial_features"](window, cfg.TARGET_FS, cfg.PDC_FREQUENCY_BANDS, cfg.CHANNEL_NAMES)
        spectral_seconds = time.perf_counter() - part_started
        part_started = time.perf_counter()
        for prefix, matrix in [("gc_roi", thresholded)] + [(f"pdc_{band}_roi", matrix.T) for band, matrix in band_pdc.items()]:
            reduced = roi["aggregate_roi_matrix"](matrix, roi_indices, roi_names)
            features.update(roi["extract_roi_features"](reduced, roi_names, prefix))
        roi_seconds = time.perf_counter() - part_started
        matrices = {"gc_raw": raw, "gc_pvalues": pv, "gc_thresholded": thresholded,
                    **{f"pdc_{b}": m for b, m in band_pdc.items()}}
        if any(m.shape != (62, 62) or not np.isfinite(m).all() for m in matrices.values()):
            raise ValueError("Invalid connectivity matrix")
        if not np.isfinite(list(features.values())).all() or len(features) != 970:
            raise ValueError("Expected 970 finite spectral + ROI features")
        if np.any(np.diag(thresholded)) or not ((pv >= 0) & (pv <= 1)).all():
            raise ValueError("Invalid GC diagonal or p-values")
        if not np.any(raw > 0):
            raise ValueError("All GC fits returned zero; inspect analyzer failures")
        if any(np.any(m < 0) or np.any(m > 1 + 1e-10) for m in band_pdc.values()):
            raise ValueError("PDC outside [0, 1]")
        audit = gc_audit(raw, pv, thresholded, cfg.GC_ALPHA)
        name = f"window_{number:03d}.npz"
        compute_elapsed_seconds = time.perf_counter() - t0
        save_started = time.perf_counter()
        np.savez_compressed(out / name, **matrices)
        rows.append({**key, **features})
        pd.DataFrame(rows).to_csv(out / "windowed_features.csv", index=False)
        save_seconds = time.perf_counter() - save_started
        manifest.append({**key, "matrix_file": name, "gc_density": density,
                         **audit, "pdc_order": int(order),
                         "gc_seconds": gc_seconds, "pdc_seconds": pdc_seconds,
                         "spectral_seconds": spectral_seconds, "roi_aggregation_seconds": roi_seconds,
                         "save_seconds": save_seconds, "elapsed_seconds": compute_elapsed_seconds})
        pd.DataFrame(manifest).to_csv(out / "window_manifest.csv", index=False)
        print(f"Window {number} passed: 970 features, {manifest[-1]['elapsed_seconds']:.1f}s", flush=True)
    timing = {
        "load_preprocess_seconds": load_preprocess_seconds,
        "spectral_seconds": sum(r["spectral_seconds"] for r in manifest),
        "gc_seconds": sum(r["gc_seconds"] for r in manifest),
        "pdc_seconds": sum(r["pdc_seconds"] for r in manifest),
        "roi_aggregation_seconds": sum(r["roi_aggregation_seconds"] for r in manifest),
        "save_seconds": sum(r["save_seconds"] for r in manifest),
        "total_seconds": time.perf_counter() - run_started,
        "per_window": [{k: r[k] for k in ["window_id", "spectral_seconds", "gc_seconds", "pdc_seconds",
                                            "roi_aggregation_seconds", "save_seconds", "elapsed_seconds"]}
                       for r in manifest],
    }
    (out / "timing_report.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    metadata.update(status="passed", completed_windows=len(rows), n_features=970,
                    total_seconds=sum(r["elapsed_seconds"] for r in manifest),
                    timing_report="timing_report.json",
                    gc_density_audit={"method": cfg.GC_THRESHOLD_METHOD,
                                      "pvalue_selection": "minimum ssr_ftest p-value across lags 1..gc_max_lag before correction",
                                      "note": "Descriptive audit only; no thresholding behavior changed."})
    report.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Output: {out}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--session-index", type=int, default=1)
    parser.add_argument("--trial", type=int, default=1)
    args = parser.parse_args()
    # Small repeated regressions run much faster without BLAS oversubscription.
    with threadpool_limits(limits=1):
        run(args.subject, args.session_index, args.trial)
