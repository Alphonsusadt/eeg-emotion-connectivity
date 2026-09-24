"""Resumable, conservative process-level runner for the frozen Phase 6.1 extractor."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
STAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import config as cfg

EXTRACTOR=Path(__file__).with_name("extract_windows.py")
OUTPUT=STAGE/"output"
SOURCE_FILES=["config.py","01_granger_causality/code/gc_analyzer.py","02_pdc/code/pdc_analyzer.py",
              "03_spectral_features/code/03_spectral_features.ipynb",
              "phase_3_feature_engineering/3.4_roi_aggregation/code/3.4_roi_aggregation.ipynb",
              "phase_6_wearable_evaluation/6.1_windowed_feature_extraction/code/extract_windows.py"]
SINGLE_THREAD_ENV=["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS",
                   "NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS"]
_BASELINE_VALIDATOR=None


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_pipeline():
    return {name:sha(ROOT/name) for name in SOURCE_FILES}


def validate_completed_artifacts(report_path):
    """Reuse the downstream final-ingestion validator; never trust status alone."""
    global _BASELINE_VALIDATOR
    if _BASELINE_VALIDATOR is None:
        path=STAGE.parent/"6.2_windowed_baseline/code/run_windowed_loso.py"
        spec=importlib.util.spec_from_file_location("batch_baseline_validator",path)
        _BASELINE_VALIDATOR=importlib.util.module_from_spec(spec); spec.loader.exec_module(_BASELINE_VALIDATOR)
    _BASELINE_VALIDATOR.read_run(Path(report_path))


def expected_trials(subjects=None,sessions=None,trials=None):
    subjects=list(range(1,16)) if subjects is None else subjects
    sessions=[1,2,3] if sessions is None else sessions
    trials=list(range(1,16)) if trials is None else trials
    if not set(subjects)<=set(range(1,16)) or not set(sessions)<=set(range(1,4)) or not set(trials)<=set(range(1,16)):
        raise ValueError("subjects=1..15, sessions=1..3, trials=1..15")
    return [(s,j,t) for s in subjects for j in sessions for t in trials]


def trial_dir(key,output_root=OUTPUT):
    s,j,t=key; date=cfg.SUBJECT_SESSIONS[s][j-1]
    return Path(output_root)/f"subject_{s:02d}"/f"session_{date}"/f"trial_{t:02d}"


def inspect_trial(key,output_root=OUTPUT,pipeline=None):
    pipeline=expected_pipeline() if pipeline is None else pipeline
    folder=trial_dir(key,output_root); reports=sorted(folder.glob("run_*/sanity_report.json"))
    valid=[]; failed=[]; incompatible=[]
    for path in reports:
        try:
            report=json.loads(path.read_text(encoding="utf-8"))
            if report.get("status")!="passed": failed.append(str(path)); continue
            reason=[]
            if report.get("pipeline_sha256")!=pipeline: reason.append("pipeline_sha256 differs from current frozen sources")
            for name in ["windowed_features.csv","window_manifest.csv","timing_report.json"]:
                if not (path.parent/name).is_file(): reason.append(f"missing {name}")
            if report.get("completed_windows")!=report.get("n_windows") or report.get("n_features")!=970:
                reason.append("incomplete window/feature count")
            if not reason:
                try: validate_completed_artifacts(path)
                except Exception as exc: reason.append(f"downstream artifact validation failed: {exc}")
            if reason: incompatible.append({"report":str(path),"reasons":reason})
            else: valid.append(path)
        except (OSError,ValueError,TypeError) as exc:
            incompatible.append({"report":str(path),"reasons":[f"malformed: {exc}"]})
    s,j,t=key; source=Path(cfg.PREPROCESS_DIR)/f"subject_{s:02d}"/f"session_{cfg.SUBJECT_SESSIONS[s][j-1]}"/f"trial_{t:02d}.npy"
    if valid: status="passed"
    elif incompatible: status="incompatible"
    elif failed: status="failed"
    else: status="missing"
    return {"subject_num":s,"session_index":j,"session":int(cfg.SUBJECT_SESSIONS[s][j-1]),"trial":t,
            "status":status,"input_available":source.is_file(),"selected_run":str(valid[-1].parent) if valid else None,
            "valid_run_count":len(valid),"superseded_valid_runs":[str(p.parent) for p in valid[:-1]],
            "failed_run_count":len(failed),"incompatible_run_count":len(incompatible),"incompatible_details":incompatible}


def inventory(keys=None,output_root=OUTPUT):
    keys=expected_trials() if keys is None else keys; pipeline=expected_pipeline()
    return pd.DataFrame([inspect_trial(k,output_root,pipeline) for k in keys])


def _init_worker():
    for name in SINGLE_THREAD_ENV: os.environ[name]="1"


def _extract_one(key):
    _init_worker()
    from threadpoolctl import threadpool_limits
    spec=importlib.util.spec_from_file_location("batch_extract_windows",EXTRACTOR)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    started=time.perf_counter()
    with threadpool_limits(limits=1): output=module.run(*key)
    timing=json.loads((output/"timing_report.json").read_text(encoding="utf-8"))
    return {"subject_num":key[0],"session_index":key[1],"trial":key[2],"status":"passed",
            "run_path":str(output.resolve()),"elapsed_seconds":time.perf_counter()-started,**timing}


def verify_parallel_gate(path=STAGE/"parallel_equivalence_report.json"):
    if not Path(path).is_file(): raise ValueError(f"workers > 1 require a passed equivalence report: {path}")
    report=json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("status")!="passed" or report.get("pipeline_sha256")!=expected_pipeline():
        raise ValueError("Parallel equivalence report is rejected or stale for current pipeline hashes")
    return report


def aggregate_timing(results):
    columns=["subject_num","session_index","trial","status","run_path","elapsed_seconds","load_preprocess_seconds",
             "spectral_seconds","gc_seconds","pdc_seconds","roi_aggregation_seconds","save_seconds","total_seconds"]
    return pd.DataFrame(results).reindex(columns=columns)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects",type=int,nargs="+")
    parser.add_argument("--sessions",type=int,nargs="+")
    parser.add_argument("--trials",type=int,nargs="+")
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--resume",action="store_true",help="Explicitly document resumable execution; passed trials are always skipped")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--audit-only",action="store_true")
    parser.add_argument("--worker-probe",action="store_true",help="Force exactly one trial in one child process to create an equivalence artifact")
    parser.add_argument("--output-root",type=Path,default=OUTPUT,help=argparse.SUPPRESS)
    parser.add_argument("--parallel-report",type=Path,default=STAGE/"parallel_equivalence_report.json")
    args=parser.parse_args(argv)
    if args.workers<1: parser.error("--workers must be >=1")
    keys=expected_trials(args.subjects,args.sessions,args.trials)
    if args.worker_probe and (len(keys)!=1 or args.dry_run or args.audit_only):
        parser.error("--worker-probe requires exactly one selected trial and performs a real, isolated probe run")
    before=inventory(keys,args.output_root); pending=[(int(r.subject_num),int(r.session_index),int(r.trial)) for _,r in before.iterrows() if r.status!="passed"]
    if args.worker_probe: pending=list(keys)  # Deliberate unique rerun for serial/worker comparison.
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"); out=Path(args.output_root)/"batch"/stamp; out.mkdir(parents=True)
    before.to_csv(out/"batch_plan.csv",index=False)
    report={"status":"audit_only" if args.audit_only else "dry_run" if args.dry_run else "running",
            "timestamp":stamp,"requested_trials":len(keys),"already_passed":int((before.status=="passed").sum()),
            "pending_trials":len(pending),"workers":args.workers,"resume":args.resume,"single_thread_env":SINGLE_THREAD_ENV,
            "pipeline_sha256":expected_pipeline(),"planned_keys":[list(k) for k in pending]}
    (out/"batch_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    if args.audit_only or args.dry_run:
        aggregate_timing([]).to_csv(out/"batch_timing_summary.csv",index=False)
        print(f"{report['status']}: passed={report['already_passed']}, pending={len(pending)}; {out}"); return 0
    if args.workers>1 and not args.worker_probe: report["parallel_equivalence"]=verify_parallel_gate(args.parallel_report)
    results=[]; failures=[]
    try:
        if args.worker_probe:
            with ProcessPoolExecutor(max_workers=1,initializer=_init_worker) as pool:
                future=pool.submit(_extract_one,pending[0])
                try: results.append(future.result())
                except Exception as exc: failures.append({"key":list(pending[0]),"error":repr(exc)})
        elif args.workers==1:
            for key in pending:
                try: results.append(_extract_one(key))
                except Exception as exc: failures.append({"key":list(key),"error":repr(exc)})
        else:
            with ProcessPoolExecutor(max_workers=args.workers,initializer=_init_worker) as pool:
                futures={pool.submit(_extract_one,key):key for key in pending}
                for future in as_completed(futures):
                    try: results.append(future.result())
                    except Exception as exc: failures.append({"key":list(futures[future]),"error":repr(exc)})
    except KeyboardInterrupt:
        report["interrupted"]=True
    aggregate_timing(results).to_csv(out/"batch_timing_summary.csv",index=False)
    report.update(status="passed" if not failures and not report.get("interrupted") else "incomplete",
                  completed_this_batch=len(results),failures=failures)
    (out/"batch_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Batch {report['status']}: completed={len(results)}, failures={len(failures)}; {out}")
    return 0 if report["status"]=="passed" else 2


if __name__=="__main__": raise SystemExit(main())
