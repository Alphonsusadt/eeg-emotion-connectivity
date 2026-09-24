"""Read-only 675-trial extraction coverage dashboard."""
import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("batch61",HERE/"run_batch_extraction.py")
batch=importlib.util.module_from_spec(spec); spec.loader.exec_module(batch)


def summarize(frame):
    counts=frame.status.value_counts().to_dict(); total=len(frame)
    return {"total_expected_trials":total,"passed":int(counts.get("passed",0)),"failed":int(counts.get("failed",0)),
            "missing":int(counts.get("missing",0)),"incompatible":int(counts.get("incompatible",0)),
            "percentage_complete":100*counts.get("passed",0)/total if total else 0,
            "all_675_complete":total==675 and counts.get("passed",0)==675,
            "per_subject":frame.groupby("subject_num").status.value_counts().unstack(fill_value=0).to_dict("index"),
            "per_session":frame.groupby("session_index").status.value_counts().unstack(fill_value=0).to_dict("index")}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--output-root",type=Path,default=batch.OUTPUT,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    frame=batch.inventory(output_root=args.output_root)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out=args.output_dir or Path(args.output_root)/"coverage"/stamp; out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out/"coverage_dashboard.csv",index=False)
    (out/"coverage_summary.json").write_text(json.dumps(summarize(frame),indent=2),encoding="utf-8")
    summary=summarize(frame)
    print(f"Coverage: {summary['passed']}/{summary['total_expected_trials']} ({summary['percentage_complete']:.2f}%); failed={summary['failed']}, missing={summary['missing']}, incompatible={summary['incompatible']}; {out}")
    return 0


if __name__=="__main__": raise SystemExit(main())
