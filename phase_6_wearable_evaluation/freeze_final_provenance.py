"""Create the canonical paper provenance manifest only after every final gate passes."""
import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PHASE=Path(__file__).resolve().parent

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def build():
    batch=load("freeze_batch",PHASE/"6.1_windowed_feature_extraction/code/run_batch_extraction.py")
    final=load("freeze_final",PHASE/"6.7_final_analysis/code/run_final_analysis.py")
    coverage=batch.inventory()
    if len(coverage)!=675 or not (coverage.status=="passed").all():
        raise ValueError(f"6.1 freeze gate failed: {(coverage.status=='passed').sum()}/675 current-compatible trials")
    prerequisite,selected=final.audit(PHASE)
    if not prerequisite["all_prerequisites_available"]:
        raise ValueError(f"Final freeze gate failed: missing={prerequisite['missing_phases']}, errors={prerequisite['errors']}")
    candidates=[]
    for path in sorted((PHASE/"6.7_final_analysis/output/final").glob("*/run_report.json")):
        report=json.loads(path.read_text(encoding="utf-8"))
        if report.get("status")=="passed" and report.get("mode")=="final": candidates.append((path,report))
    if not candidates: raise ValueError("Final freeze gate failed: no passed final 6.7 run")
    path67,run67=candidates[-1]
    if run67.get("status")!="passed" or run67.get("mode")!="final": raise ValueError("6.7 is not passed final")
    commit=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
    code={}
    code["6.1"]={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in sorted((PHASE/"6.1_windowed_feature_extraction/code").glob("*.py"))}
    for phase,entry in final.PHASES.items():
        folder=PHASE/entry[0]/"code"
        code[phase]={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in sorted(folder.glob("*.py"))}
    code["6.7"]={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in sorted((PHASE/"6.7_final_analysis/code").glob("*.py"))}
    selected61=[]
    for _,row in coverage.sort_values(["subject_num","session_index","trial"]).iterrows():
        run=Path(row.selected_run)
        selected61.append({"subject_num":int(row.subject_num),"session_index":int(row.session_index),"trial":int(row.trial),
                           "run_path":str(run.resolve()),"sanity_report_sha256":sha(run/"sanity_report.json"),
                           "features_sha256":sha(run/"windowed_features.csv")})
    reports={p:s["report"] for p,s in selected.items()}
    feature_hash=run67.get("feature_schema_hash") or reports["6.2"].get("feature_schema_hash") or final._feature_fingerprint(reports["6.2"])
    final_paths={p:s["path"] for p,s in selected.items()}; final_paths["6.7"]=str(path67.parent.resolve())
    return {"schema_version":"phase6-paper-freeze-v1","repo_commit_sha":commit,"config_sha256":sha(ROOT/"config.py"),
            "code_hashes_per_phase":code,"selected_6_1_runs":selected61,
            "final_run_paths":final_paths,"feature_schema_hash":feature_hash,
            "historical_tuning_parameter_hash":sha(ROOT/"phase_4_classification/4.6_hyperparameter_tuning/output/tuning_best_params.csv"),
            "freeze_assertions":{"all_675_trials_validated":True,"all_final_runs_passed":True,
                                 "no_pilot_audit_synthetic_evidence":True,"phase_6_7_consistency_gate_passed":True}}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path,default=PHASE/"final_provenance_manifest.json"); args=parser.parse_args(argv)
    try: manifest=build()
    except (ValueError,OSError,KeyError,subprocess.CalledProcessError) as exc:
        print(f"Rejected; no manifest written: {exc}"); return 2
    args.output.write_text(json.dumps(manifest,indent=2),encoding="utf-8"); print(f"Frozen provenance: {args.output}"); return 0

if __name__=="__main__": raise SystemExit(main())
