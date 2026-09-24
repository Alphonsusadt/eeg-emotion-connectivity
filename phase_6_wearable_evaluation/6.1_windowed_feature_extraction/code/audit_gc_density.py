"""Read-only audit of saved Phase 6.1 GC density and active threshold behavior."""
import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
STAGE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("gc_density_audit",ROOT/"01_granger_causality/code/gc_analyzer.py")
gc=importlib.util.module_from_spec(spec); spec.loader.exec_module(gc)


def audit_run(run):
    run=Path(run); report=json.loads((run/"sanity_report.json").read_text(encoding="utf-8")); rows=[]
    for path in sorted(run.glob("window_*.npz")):
        with np.load(path,allow_pickle=False) as data:
            raw,p,stored=data["gc_raw"],data["gc_pvalues"],data["gc_thresholded"]
        reproduced=gc.apply_threshold(raw,p,report["gc_threshold"],report["gc_alpha"])
        if not np.array_equal(reproduced,stored):
            raise ValueError(f"Suspected threshold implementation/provenance mismatch: {path}")
        mask=~np.eye(raw.shape[0],dtype=bool); values=p[mask]; possible=len(values); rejected=np.count_nonzero(stored[mask])
        rows.append({"run_path":str(run.resolve()),"window_file":path.name,"window_id":int(path.stem.split("_")[-1]),
                     "gc_raw_nonzero_density":np.count_nonzero(raw[mask])/possible,
                     "gc_thresholded_density":rejected/possible,"fdr_rejected_edge_count":rejected,
                     "possible_edges":possible,"p_min":values.min(),"p_q01":np.quantile(values,.01),
                     "p_q05":np.quantile(values,.05),"p_median":np.median(values),"p_q95":np.quantile(values,.95),
                     "p_max":values.max(),"uncorrected_p_lt_alpha":np.count_nonzero(values<report["gc_alpha"])})
    if not rows: raise ValueError(f"No saved window matrices: {run}")
    return pd.DataFrame(rows),report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args(argv); table,report=audit_run(args.run)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"); out=args.output_dir or STAGE/"output"/"gc_density_audit"/stamp; out.mkdir(parents=True,exist_ok=True)
    table.to_csv(out/"gc_density_audit.csv",index=False)
    summary={"status":"passed","run_path":str(args.run.resolve()),"active_method":report["gc_threshold"],
             "active_behavior":"minimum ssr_ftest p-value across configured lags, then one BH-FDR across directed off-diagonal edges",
             "mean_raw_nonzero_density":table.gc_raw_nonzero_density.mean(),
             "mean_thresholded_density":table.gc_thresholded_density.mean(),
             "total_fdr_rejected_edges":int(table.fdr_rejected_edge_count.sum()),
             "historical_documentation":"README target <30%; older percentile outputs commonly ~0.1002",
             "interpretation":"Stored matrices exactly reproduce the active implementation. High density is method behavior, not evidence of a storage/threshold application bug. Preserve and review scientifically; do not silently change."}
    (out/"gc_density_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(f"GC density audit passed: mean={summary['mean_thresholded_density']:.6f}; {out}"); return 0


if __name__=="__main__": raise SystemExit(main())
