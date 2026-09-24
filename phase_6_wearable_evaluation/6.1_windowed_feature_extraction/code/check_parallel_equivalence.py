"""Compare serial and process-worker artifacts before enabling parallel extraction."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


class EquivalenceError(ValueError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_runs(serial, worker, tolerance=1e-10):
    serial, worker = Path(serial), Path(worker)
    required = ["sanity_report.json", "windowed_features.csv", "window_manifest.csv"]
    for root in (serial, worker):
        missing = [name for name in required if not (root/name).is_file()]
        if missing: raise EquivalenceError(f"{root}: missing {missing}")
    sr, wr = load_json(serial/"sanity_report.json"), load_json(worker/"sanity_report.json")
    if sr.get("status") != "passed" or wr.get("status") != "passed":
        raise EquivalenceError("Both runs must be passed")
    identity = ["source_sha256", "pipeline_sha256", "fs", "window_seconds", "overlap_samples",
                "tail_policy", "input_shape", "n_windows", "completed_windows", "n_features", "channels",
                "gc_max_lag", "gc_threshold", "gc_alpha", "pdc_max_order", "pdc_min_order", "bands",
                "matrix_orientation", "normalization"]
    mismatch = [key for key in identity if sr.get(key) != wr.get(key)]
    if mismatch: raise EquivalenceError(f"Metadata/provenance mismatch: {mismatch}")
    a, b = pd.read_csv(serial/"windowed_features.csv"), pd.read_csv(worker/"windowed_features.csv")
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        raise EquivalenceError("Feature CSV schema/row mismatch")
    numeric = a.select_dtypes(include=[np.number]).columns
    if not np.allclose(a[numeric], b[numeric], rtol=0, atol=tolerance, equal_nan=False):
        delta = np.max(np.abs(a[numeric].to_numpy(float)-b[numeric].to_numpy(float)))
        raise EquivalenceError(f"Feature values differ; max_abs={delta}")
    other = [c for c in a.columns if c not in numeric]
    if other and not a[other].equals(b[other]): raise EquivalenceError("Feature metadata differs")
    am, bm = pd.read_csv(serial/"window_manifest.csv"), pd.read_csv(worker/"window_manifest.csv")
    stable = [c for c in am.columns if not c.endswith("_seconds") and c != "elapsed_seconds"]
    if stable != [c for c in bm.columns if not c.endswith("_seconds") and c != "elapsed_seconds"]:
        raise EquivalenceError("Manifest schema differs")
    for column in stable:
        if pd.api.types.is_numeric_dtype(am[column]):
            if not np.allclose(am[column], bm[column], rtol=0, atol=tolerance):
                raise EquivalenceError(f"Manifest differs: {column}")
        elif not am[column].equals(bm[column]): raise EquivalenceError(f"Manifest differs: {column}")
    matrix_hashes = {}
    for name in am.matrix_file:
        with np.load(serial/name, allow_pickle=False) as left, np.load(worker/name, allow_pickle=False) as right:
            if set(left.files) != set(right.files): raise EquivalenceError(f"Matrix keys differ: {name}")
            for key in left.files:
                if not np.allclose(left[key], right[key], rtol=0, atol=tolerance):
                    delta=float(np.max(np.abs(left[key]-right[key])))
                    raise EquivalenceError(f"{name}/{key} differs; max_abs={delta}")
        matrix_hashes[name] = {"serial": sha(serial/name), "worker": sha(worker/name)}
    return {"status":"passed", "tolerance":tolerance, "serial_run":str(serial.resolve()),
            "worker_run":str(worker.resolve()), "source_sha256":sr["source_sha256"],
            "pipeline_sha256":sr["pipeline_sha256"], "n_windows":len(a),
            "feature_csv_sha256":{"serial":sha(serial/"windowed_features.csv"),"worker":sha(worker/"windowed_features.csv")},
            "matrix_file_hashes":matrix_hashes}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-run",type=Path,required=True)
    parser.add_argument("--worker-run",type=Path,required=True)
    parser.add_argument("--tolerance",type=float,default=1e-10)
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"parallel_equivalence_report.json")
    args=parser.parse_args(argv)
    try:
        report=compare_runs(args.serial_run,args.worker_run,args.tolerance)
    except (EquivalenceError,OSError,ValueError,KeyError) as exc:
        report={"status":"rejected","reason":str(exc),"tolerance":args.tolerance}
        args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(f"Rejected: {exc}"); return 2
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Parallel equivalence passed: {args.output}"); return 0


if __name__=="__main__": raise SystemExit(main())
