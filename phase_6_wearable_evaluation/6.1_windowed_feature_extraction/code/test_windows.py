"""Synthetic execution/readiness checks; never run expensive real extraction."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
from extract_windows import window_bounds, notebook_definitions, ROI
import extract_windows as extract

HERE=Path(__file__).resolve().parent
def load(name, filename):
    spec=importlib.util.spec_from_file_location(name,HERE/filename); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
batch=load("batch_test61","run_batch_extraction.py")
coverage=load("coverage_test61","check_extraction_coverage.py")
equivalence=load("equivalence_test61","check_parallel_equivalence.py")


class WindowTests(unittest.TestCase):
    def test_boundaries_and_tail(self):
        self.assertEqual(window_bounds(5999), [])
        self.assertEqual(window_bounds(6000), [(0, 6000)])
        self.assertEqual(window_bounds(12000), [(0, 6000), (6000, 12000)])
        self.assertEqual(window_bounds(23501), [(0, 6000), (6000, 12000), (12000, 18000)])

    def test_roi_direction_for_pdc(self):
        ns = notebook_definitions(ROI, ["aggregate_roi_matrix", "extract_roi_features"], {"np": np})
        # Raw PDC row B, column A encodes A -> B.
        raw = np.array([[0.8, 0.0], [0.6, 1.0]])
        reduced = ns["aggregate_roi_matrix"](raw.T, {"A": [0], "B": [1]}, ["A", "B"])
        features = ns["extract_roi_features"](reduced, ["A", "B"], "pdc")
        self.assertEqual(features["pdc_A_to_B"], 0.6)
        self.assertEqual(features["pdc_B_to_A"], 0.0)
        self.assertEqual(features["pdc_out_strength_A"], 0.6)

    def fake_run(self, root, name):
        source=Path(root)/"input"/"subject_01"/f"session_{extract.cfg.SUBJECT_SESSIONS[1][0]}"/"trial_01.npy"
        source.parent.mkdir(parents=True,exist_ok=True); np.save(source,np.arange(62*6000,dtype=float).reshape(62,6000)%101)
        spectral={f"spec_{i}":float(i) for i in range(620)}
        def aggregate(matrix,indices,names): return matrix[:6,:6]
        def roi_features(matrix,names,prefix): return {f"{prefix}_f{i}":float(i) for i in range(50)}
        gc=SimpleNamespace(analyze_trial=lambda window:(np.ones((62,62))-np.eye(62),np.full((62,62),.01),np.ones((62,62))-np.eye(62),1.0))
        pdc=SimpleNamespace(analyze_trial=lambda window:({b:np.full((62,62),.1) for b in list(extract.cfg.PDC_FREQUENCY_BANDS)+["broadband"]},3))
        modules=[gc,pdc]
        def loader(*args): return modules.pop(0)
        def notebook(path,names,namespace):
            if "extract_trial_features" in names: return {"extract_trial_features":lambda *args:spectral.copy()}
            return {"ROI_GROUPS":{f"R{i}":[extract.cfg.CHANNEL_NAMES[i]] for i in range(6)},
                    "aggregate_roi_matrix":aggregate,"extract_roi_features":roi_features}
        stage=Path(root)/name
        with mock.patch.object(extract,"STAGE",stage),mock.patch.object(extract.cfg,"PREPROCESS_DIR",str(Path(root)/"input")), \
             mock.patch.object(extract,"load_module",side_effect=loader),mock.patch.object(extract,"notebook_definitions",side_effect=notebook):
            return extract.run(1,1,1)

    def test_original_single_trial_behavior_preserved_with_timing_and_gc_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=self.fake_run(tmp,"serial")
            report=json.loads((out/"sanity_report.json").read_text()); self.assertEqual(report["status"],"passed")
            self.assertEqual(report["completed_windows"],1); self.assertEqual(report["n_features"],970)
            self.assertTrue((out/"timing_report.json").is_file())
            features=pd.read_csv(out/"windowed_features.csv"); self.assertEqual(features.shape,(1,982))
            manifest=pd.read_csv(out/"window_manifest.csv"); self.assertEqual(manifest.loc[0,"gc_fdr_rejected_edges"],62*61)

    def test_serial_and_worker_artifacts_equivalent(self):
        with tempfile.TemporaryDirectory() as tmp:
            serial=self.fake_run(tmp,"serial"); worker=self.fake_run(tmp,"worker")
            result=equivalence.compare_runs(serial,worker,1e-10)
            self.assertEqual(result["status"],"passed"); self.assertEqual(result["n_windows"],1)


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.validator=mock.patch.object(batch,"validate_completed_artifacts",return_value=None)
        self.validator.start()

    def tearDown(self):
        self.validator.stop()

    def make_report(self,root,key,status="passed",pipeline=None,stamp="run_1"):
        folder=batch.trial_dir(key,root)/stamp; folder.mkdir(parents=True,exist_ok=True)
        info={"status":status,"pipeline_sha256":batch.expected_pipeline(),"completed_windows":1,"n_windows":1,"n_features":970}
        if pipeline is not None: info["pipeline_sha256"]=pipeline
        (folder/"sanity_report.json").write_text(json.dumps(info))
        if status=="passed":
            pd.DataFrame([{"x":1}]).to_csv(folder/"windowed_features.csv",index=False)
            pd.DataFrame([{"x":1}]).to_csv(folder/"window_manifest.csv",index=False)
            (folder/"timing_report.json").write_text(json.dumps({"total_seconds":1}))
        return folder

    def test_completed_trial_skipped_and_duplicate_not_duplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_report(tmp,(1,1,1),stamp="run_1"); latest=self.make_report(tmp,(1,1,1),stamp="run_2")
            row=batch.inspect_trial((1,1,1),tmp)
            self.assertEqual(row["status"],"passed"); self.assertEqual(row["valid_run_count"],2)
            self.assertEqual(row["selected_run"],str(latest)); self.assertEqual(len(row["superseded_valid_runs"]),1)
            with mock.patch.object(batch,"_extract_one",side_effect=AssertionError("must skip")):
                self.assertEqual(batch.main(["--subjects","1","--sessions","1","--trials","1","--dry-run","--output-root",tmp]),0)

    def test_failed_and_interrupted_trials_are_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_report(tmp,(1,1,1),status="running")
            self.assertEqual(batch.inspect_trial((1,1,1),tmp)["status"],"failed")
            self.make_report(tmp,(1,1,1),status="passed",stamp="run_2")
            self.assertEqual(batch.inspect_trial((1,1,1),tmp)["status"],"passed")

    def test_invalid_provenance_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_report(tmp,(1,1,1),pipeline={"wrong":"hash"})
            row=batch.inspect_trial((1,1,1),tmp); self.assertEqual(row["status"],"incompatible")

    def test_coverage_count_exact_675(self):
        rows=[]
        for key in batch.expected_trials():
            rows.append({"subject_num":key[0],"session_index":key[1],"trial":key[2],"status":"missing"})
        rows[0]["status"]="passed"; rows[1]["status"]="failed"; rows[2]["status"]="incompatible"
        summary=coverage.summarize(pd.DataFrame(rows))
        self.assertEqual(summary["total_expected_trials"],675); self.assertEqual(summary["passed"],1)
        self.assertEqual(summary["missing"],672)

    def test_dry_run_and_audit_only_never_extract(self):
        for flag in ["--dry-run","--audit-only"]:
            with self.subTest(flag=flag),tempfile.TemporaryDirectory() as tmp, \
                 mock.patch.object(batch,"_extract_one",side_effect=AssertionError("extraction called")):
                self.assertEqual(batch.main(["--subjects","1","--sessions","1","--trials","1",flag,"--output-root",tmp]),0)

    def test_parallel_requires_current_equivalence_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=Path(tmp)/"parallel.json"; report.write_text(json.dumps({"status":"passed","pipeline_sha256":{"stale":1}}))
            with self.assertRaisesRegex(ValueError,"stale"):
                batch.verify_parallel_gate(report)

    def test_worker_probe_is_restricted_to_one_trial(self):
        with tempfile.TemporaryDirectory() as tmp,self.assertRaises(SystemExit):
            batch.main(["--subjects","1","--sessions","1","--trials","1","2","--worker-probe","--output-root",tmp])


if __name__ == "__main__":
    unittest.main()
