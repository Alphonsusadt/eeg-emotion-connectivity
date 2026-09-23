"""Boundary and direction checks independent of the expensive EEG sanity run."""
import unittest
import numpy as np
from extract_windows import window_bounds, notebook_definitions, ROI


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


if __name__ == "__main__":
    unittest.main()
