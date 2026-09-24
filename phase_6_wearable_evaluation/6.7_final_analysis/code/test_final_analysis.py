"""Synthetic-only tests for the Phase 6.7 read/validate/integrate layer."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("final67", HERE / "run_final_analysis.py")
fa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fa)


def coverage():
    return {"complete_subjects": list(range(1, 16)), "selected_runs": [
        {"subject_num": s, "session": j, "trial": t, "report_sha256": f"r{s}-{j}-{t}", "csv_sha256": f"c{s}-{j}-{t}"}
        for s in range(1, 16) for j in range(1, 4) for t in range(1, 16)]}


def base_rows():
    rows=[]
    for di,d in enumerate(fa.DATASETS):
        for s in range(1,16):
            value=.45+.02*di+.002*s
            rows.append({"mode":"final","dataset":d,"subject_num":s,"subject_id":f"S{s:02d}",
                         "accuracy":value,"macro_f1":value-.01,"weighted_f1":value-.005})
    return pd.DataFrame(rows)


def phase_tables():
    p62=base_rows()
    p63=[]
    effects={"0":0,"1":-.01,"2":0,"5":.01,"10":.015,"full":.02}
    for _,r in p62.iterrows():
        for c,e in effects.items():
            p63.append({**r.to_dict(),"condition":c,"accuracy":r.accuracy+e,"macro_f1":r.macro_f1+e,
                        "weighted_f1":r.weighted_f1+e,"matched_gain_pp":100*e,
                        "matched_gap_to_full_pp":100*(.02-e),"matched_recovery_pct":100*e/.02 if c!="0" else 0})
    p63=pd.DataFrame(p63)
    p64=[]
    for _,r in p62.iterrows():
        for c,e in [("zero_cal",-.03),("session1_cal",-.02)]:
            p64.append({**r.to_dict(),"condition":c,"accuracy":r.accuracy+e,"macro_f1":r.macro_f1+e,"weighted_f1":r.weighted_f1+e})
    p64=pd.DataFrame(p64)
    p65=[]
    for _,r in p62.iterrows():
        for c,e in [("zero_cal",-.03),("fixed_session1",-.02),("streaming_running",-.025),("streaming_ema",-.02)]:
            p65.append({**r.to_dict(),"condition":c,"accuracy":r.accuracy+e,"macro_f1":r.macro_f1+e,
                        "weighted_f1":r.weighted_f1+e,"gain_vs_zero_pp":100*(e+.03),"gain_vs_fixed_pp":100*(e+.02)})
    p65=pd.DataFrame(p65)
    p66=[]
    for i in range(12):
        value=-.2+.08*i
        p66.append({"mode":"final","analysis":"raw_matched_session_units","normalization":"raw","feature_name":f"f{i}",
                    "feature_family":"Spectral" if i<6 else "Connectivity","method":"PSD_power" if i<6 else "PDC_ROI",
                    "band":"alpha" if i%2 else "theta","ICC2_1":value,"ICC3_1":value+.05,
                    "ICC2_1_status":"ok","ICC3_1_status":"ok","n_subjects":15,"n_sessions":3})
    p66=pd.DataFrame(p66)
    pair=pd.DataFrame([{"feature_name":f,"session_pair":pair,"Pearson_r":.2,"Spearman_rho":.2}
                       for f in p66.feature_name for pair in ["S1_S2","S1_S3","S2_S3"]])
    family=pd.DataFrame([{"feature_family":"Spectral","method":"PSD_power","band":"alpha","icc_type":"ICC2_1","n_features":6,"median_ICC":.2}])
    return p62,p63,p64,p65,p66,family,pair


def write_fixture(root, stamp="20260101T000000Z", report_changes=None):
    p62,p63,p64,p65,p66,family,pair=phase_tables()
    mapping={
        "6.2":{"windowed_loso_per_subject.csv":p62,"windowed_loso_summary.csv":p62.groupby("dataset").accuracy.mean().reset_index(),"windowed_predictions.csv":p62},
        "6.3":{"calibration_per_subject.csv":p63,"calibration_summary.csv":p63.groupby(["dataset","condition"]).accuracy.mean().reset_index(),"calibration_predictions.csv":p63,"calibration_curve.csv":p63.groupby(["dataset","condition"]).accuracy.mean().reset_index()},
        "6.4":{"cross_session_per_subject.csv":p64,"cross_session_summary.csv":p64.groupby(["dataset","condition"]).accuracy.mean().reset_index(),"cross_session_predictions.csv":p64},
        "6.5":{"streaming_per_subject.csv":p65,"streaming_summary.csv":p65.groupby(["dataset","condition"]).accuracy.mean().reset_index(),"streaming_predictions.csv":p65,"streaming_cumulative_curve.csv":p65},
        "6.6":{"reliability_per_feature.csv":p66,"reliability_family_summary.csv":family,"reliability_pairwise.csv":pair},
    }
    for phase,(folder,_) in fa.PHASES.items():
        run=Path(root)/folder/"output"/"final"/stamp; run.mkdir(parents=True)
        report={"status":"passed","mode":"final","audit_only":False,"timestamp":stamp,"feature_schema_hash":"schema-v1",
                "parameter_hash":"params-v1" if phase!="6.6" else None,"protocol":"primary" if phase in ["6.4","6.5"] else phase,
                "code_hash":f"code-{phase}","config_hash":"config-v1"}
        if report_changes and phase in report_changes: report.update(report_changes[phase])
        (run/"run_report.json").write_text(json.dumps(report),encoding="utf-8")
        (run/"coverage_report.json").write_text(json.dumps(coverage()),encoding="utf-8")
        for name,frame in mapping[phase].items(): frame.to_csv(run/name,index=False)
    return mapping


class FinalAnalysisTests(unittest.TestCase):
    def selected(self, root):
        report, selected=fa.audit(Path(root)); self.assertTrue(report["all_prerequisites_available"]); return selected

    def test_latest_compatible_passed_final_selected_and_nonfinal_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(tmp,"20260101T000000Z"); write_fixture(tmp,"20260102T000000Z")
            pilot=Path(tmp)/fa.PHASES["6.2"][0]/"output"/"pilot"/"20990101"; pilot.mkdir(parents=True)
            (pilot/"run_report.json").write_text(json.dumps({"status":"passed","mode":"pilot"}))
            choice,candidates=fa.select_latest_run(Path(tmp),"6.2")
            self.assertTrue(choice["path"].endswith("20260102T000000Z")); self.assertEqual(len(candidates),2)

    def test_synthetic_audit_and_pilot_never_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/fa.PHASES["6.2"][0]/"output"
            for mode in ["audit","pilot","synthetic"]:
                run=folder/mode/"x"; run.mkdir(parents=True); (run/"run_report.json").write_text(json.dumps({"status":"passed","mode":mode}))
            choice,candidates=fa.select_latest_run(Path(tmp),"6.2")
            self.assertIsNone(choice); self.assertEqual(candidates,[])

    def test_missing_phase_rejects_final_but_audit_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(tmp)
            target=Path(tmp)/fa.PHASES["6.6"][0]/"output"/"final"/"20260101T000000Z"/"run_report.json"
            target.unlink()
            out=Path(tmp)/"out"; figs=Path(tmp)/"figs"
            self.assertEqual(fa.main(["--audit-only","--phase-root",tmp,"--output-root",str(out),"--figures-root",str(figs)]),0)
            report=json.loads(next(out.glob("audit/*/prerequisite_report.json")).read_text()); self.assertIn("6.6",report["missing_phases"])
            self.assertEqual(fa.main(["--require-complete","--phase-root",tmp,"--output-root",str(out),"--figures-root",str(figs)]),2)
            self.assertFalse(list(out.glob("final/*/final_baseline_table.csv")))

    def test_incompatible_provenance_rejects_ambiguous_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(tmp,"20260101T000000Z"); write_fixture(tmp,"20260102T000000Z",{"6.2":{"parameter_hash":"changed"}})
            with self.assertRaisesRegex(fa.IntegrationError,"ambiguous incompatible"):
                fa.select_latest_run(Path(tmp),"6.2")

    def test_duplicate_subject_condition_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(tmp); selected=self.selected(tmp)
            path=Path(selected["6.5"]["path"])/"streaming_per_subject.csv"; frame=pd.read_csv(path); pd.concat([frame,frame.iloc[[0]]]).to_csv(path,index=False)
            with self.assertRaisesRegex(fa.IntegrationError,"duplicate"):
                fa.validate_inputs(selected,1e-10)

    def test_nonfinite_metric_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(tmp); selected=self.selected(tmp)
            path=Path(selected["6.2"]["path"])/"windowed_loso_per_subject.csv"; frame=pd.read_csv(path); frame.loc[0,"accuracy"]=np.inf; frame.to_csv(path,index=False)
            with self.assertRaisesRegex(fa.IntegrationError,"non-finite"):
                fa.validate_inputs(selected,1e-10)

    def test_cross_phase_mismatch_rejects_with_exact_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(tmp); selected=self.selected(tmp)
            path=Path(selected["6.5"]["path"])/"streaming_per_subject.csv"; frame=pd.read_csv(path)
            frame.loc[(frame.dataset=="Spectral")&(frame.subject_num==1)&(frame.condition=="fixed_session1"),"accuracy"]+=.01; frame.to_csv(path,index=False)
            with self.assertRaisesRegex(fa.IntegrationError,"fixed_session1 mismatch"):
                fa.validate_inputs(selected,1e-10)

    def test_exact_values_preserved_and_negative_results_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(tmp); selected=self.selected(tmp); tables=fa.validate_inputs(selected,1e-10); outputs=fa.build_tables(tables)
            source=tables["6.5"]["streaming_per_subject.csv"]
            expected=source[(source.dataset=="Spectral")&(source.condition=="streaming_running")].accuracy.mean()
            got=outputs[3][(outputs[3].Dataset=="Spectral")&(outputs[3].Condition=="streaming_running")]["Accuracy mean"].iloc[0]
            self.assertEqual(got,expected)
            summary=fa.paper_summary(outputs,fa.planned_tests(tables))
            self.assertIn("-0.50",summary); self.assertNotIn("online adaptation was superior",summary.lower())

    def test_no_experiment_runner_or_model_functions_imported(self):
        source=(HERE/"run_final_analysis.py").read_text(encoding="utf-8")
        forbidden=["run_windowed_loso","run_calibration_length","run_cross_session","run_streaming_normalization","run_session_reliability",".fit(","GridSearchCV","bootstrap_ci(","extract_features"]
        for token in forbidden: self.assertNotIn(token,source)
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(tmp); selected=self.selected(tmp)
            with mock.patch.object(fa,"read_csv",wraps=fa.read_csv) as reader:
                fa.validate_inputs(selected,1e-10); self.assertTrue(reader.called)

    def test_paired_test_intersects_exact_ids_and_rejects_unmatched(self):
        frame=phase_tables()[2]
        row=fa.paired_comparison(frame,"zero_cal","session1_cal","test","Spectral")
        self.assertEqual(row["n_pairs"],15); self.assertEqual(row["paired_subject_ids"].split(";")[0],"1")
        bad=frame[~((frame.dataset=="Spectral")&(frame.subject_num==15)&(frame.condition=="session1_cal"))]
        with self.assertRaisesRegex(fa.IntegrationError,"paired subjects differ"):
            fa.paired_comparison(bad,"zero_cal","session1_cal","test","Spectral")

    def test_calibration_test_uses_matched_row_gain(self):
        frame=phase_tables()[1]
        chosen=frame[(frame.dataset=="Spectral")&(frame.condition=="5")].index
        frame.loc[chosen,"matched_gain_pp"]=[-2+i/10 for i in range(15)]
        # Raw condition accuracy is intentionally unrelated; the reported effect must
        # preserve the explicitly supplied same-row gain values.
        frame.loc[chosen,"accuracy"]=.99
        row=fa.matched_gain_comparison(frame,"5","Spectral")
        self.assertAlmostEqual(row["mean_paired_difference_pp"],frame.loc[chosen,"matched_gain_pp"].mean())

    def test_holm_known_values(self):
        np.testing.assert_allclose(fa.holm_adjust([.01,.04,.03]),[.03,.06,.06])

    def test_full_integration_outputs_without_recomputation(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(tmp); out=Path(tmp)/"integrated"; figs=Path(tmp)/"figures"
            self.assertEqual(fa.main(["--require-complete","--phase-root",tmp,"--output-root",str(out),"--figures-root",str(figs)]),0)
            run=next((out/"final").iterdir())
            expected=["final_baseline_table.csv","final_calibration_table.csv","final_cross_session_table.csv","final_streaming_table.csv","final_reliability_table.csv","planned_pairwise_tests.csv","historical_phase5_reference.csv","upstream_provenance.json","paper_results_summary.md","run_report.json","prerequisite_report.json"]
            for name in expected: self.assertTrue((run/name).is_file(),name)
            for name in ["fig_main_baseline.png","fig_calibration_curve.png","fig_cross_session.png","fig_streaming_adaptation.png","fig_reliability.png"]:
                self.assertTrue(next((figs/"final").glob(f"*/{name}"),None),name)
            historical=pd.read_csv(run/"historical_phase5_reference.csv")
            self.assertTrue((historical.analysis_level=="trial").all()); self.assertAlmostEqual(historical.iloc[0].accuracy_mean_percent,74.22)


if __name__ == "__main__":
    unittest.main(verbosity=2)
