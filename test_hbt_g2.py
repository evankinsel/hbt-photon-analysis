"""HBT regressions: python -m unittest -v test_hbt_g2.py"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import hbt_nist_acquisition as hbt


class HbtCorrelationTests(unittest.TestCase):
    def settings(self, **changes):
        return {**hbt.g2_settings("test"), **changes}

    def test_all_pairs_edges_and_delay_against_brute_force(self):
        a = np.array([0, 10, 20, 100], dtype=np.int64)
        b = np.array([0, 2, 8, 11, 14, 22, 100], dtype=np.int64)
        edges = np.arange(-15, 26, 10, dtype=np.int64)
        for delay in (-3, 0, 3):
            differences = b[None, :] - a[:, None] - delay
            expected = [np.count_nonzero((differences >= lo) & (differences < hi))
                        for lo, hi in zip(edges[:-1], edges[1:])]
            actual = hbt.g2_pair_counts(a, b, edges, delay, 1000)
            np.testing.assert_array_equal(actual, expected)
        # A pair exactly on the final right edge is excluded, even in the final bin.
        np.testing.assert_array_equal(hbt.g2_pair_counts(np.array([0]), np.array([5]),
                                                       np.array([-5, 5]), 0, 100), [0])

    def test_dense_expansion_is_chunked_and_budget_is_not_silent(self):
        a = np.arange(600, dtype=np.int64)
        edges = np.arange(-610, 611, 20, dtype=np.int64)
        counts = hbt.g2_pair_counts(a, a, edges, 0, 400000)
        self.assertEqual(counts.sum(), 360000)
        with self.assertRaisesRegex(ValueError, "pair limit"):
            hbt.g2_pair_counts(a, a, edges, 0, 1000)

    def test_overlap_integral_and_delay_correction(self):
        # T=10; integral under the triangular overlap function.
        edges = np.array([-10, -5, 0, 5, 10])
        np.testing.assert_allclose(hbt.g2_overlap_area(edges, 10), [12.5, 37.5, 37.5, 12.5])
        np.testing.assert_allclose(hbt.g2_overlap_area(edges, 10, 5), [37.5, 37.5, 12.5, 0])
        np.testing.assert_allclose(hbt.g2_overlap_area(np.array([-1, 1]), 10), [19])
        # Wide exposure, narrow bins: avoid catastrophic cancellation.
        np.testing.assert_allclose(hbt.g2_overlap_area(np.array([-500, 500]), 10**13),
                                   [1000 * 10**13 - 250000], rtol=1e-14)

    def test_known_correlated_peak_and_calibrated_delay(self):
        a = np.arange(100, 9900, 100, dtype=np.int64)
        b = a + 20
        settings = self.settings(bin_width_ps=10, half_bins=4, delay_ps=20)
        report, columns = hbt.calculate_g2(a, b, settings, window_ps=[0, 10000])
        self.assertEqual(report["central_bin"]["counts"], len(a))
        self.assertEqual(int(columns["lag_ps"][np.argmax(columns["coincidences"])]), 0)
        uncorrected, uncorrected_cols = hbt.calculate_g2(a, b, {**settings, "delay_ps": 0}, [0, 10000])
        self.assertEqual(int(uncorrected_cols["lag_ps"][np.argmax(uncorrected_cols["coincidences"])]), 20)
        self.assertEqual(uncorrected["central_bin"]["counts"], 0)
        self.assertGreater(uncorrected["central_bin"]["approximate_poisson_95_interval"][1], 0)
        expected_center = (len(a) / 10000) ** 2 * 10 * (10000 - 20)
        self.assertAlmostEqual(report["central_bin"]["expected_accidental_counts"], expected_center)

    def test_independent_uniform_arrivals_have_unit_baseline(self):
        rng = np.random.default_rng(41028)
        duration = 10**10
        a = np.unique(rng.integers(0, duration, 80000, dtype=np.int64))
        b = np.unique(rng.integers(0, duration, 80000, dtype=np.int64))
        settings = self.settings(bin_width_ps=10000, half_bins=20)
        report, columns = hbt.calculate_g2(a, b, settings, [0, duration])
        self.assertLess(abs(float(columns["g2"].mean()) - 1), .04)
        self.assertEqual(report["window_source"], "explicit_capture_window")
        self.assertTrue(np.all(columns["poisson95_upper"] > columns["poisson95_lower"]))

    def test_large_absolute_timestamps_and_no_input_mutation(self):
        a = np.arange(100, 9900, 100, dtype=np.int64)
        b = a + 20
        shift = 8_000_000_000_000_000_000
        settings = self.settings(bin_width_ps=10, half_bins=4, delay_ps=20)
        _, base = hbt.calculate_g2(a, b, settings)
        aa, bb = a + shift, b + shift
        before_a, before_b = aa.copy(), bb.copy()
        report, moved = hbt.calculate_g2(aa, bb, settings)
        np.testing.assert_array_equal(moved["coincidences"], base["coincidences"])
        np.testing.assert_allclose(moved["g2"], base["g2"])
        np.testing.assert_array_equal(aa, before_a)
        np.testing.assert_array_equal(bb, before_b)
        self.assertEqual(report["window_source"], "inferred_intersection_of_observed_event_spans")

    def test_invalid_and_insufficient_data(self):
        settings = self.settings(bin_width_ps=2, half_bins=1)
        for a, b in (([1], [1, 2]), ([1, 1, 3], [1, 2, 3]),
                     ([1, 2, 3], [10, 11, 12]), ([1., 2., 3.], [1, 2, 3])):
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                hbt.calculate_g2(a, b, settings)
        for changes in ({"bin_width_ps": 3}, {"delay_ps": 1.5}, {"half_bins": 0},
                        {"max_pairs": 0}, {"half_bins": 100001}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                hbt.validate_g2_settings({**settings, **changes})

    def make_run(self, folder):
        a = np.arange(10000, 990000, 10000, dtype=np.int64)
        b = a + 2000
        for channel, values in ((2, a), (3, b)):
            np.save(folder / f"raw_ch{channel}.npy", values)
        info = {"valid": True, "status": "complete", "channels": [2, 3],
                "missed_events": 0, "non_photon_tags": 0, "hardware_overflows": 0,
                "raw_events": {"2": len(a), "3": len(b)}}
        hbt.save_json(folder / "acquisition.json", info)
        return info

    def test_exports_saved_settings_and_does_not_touch_nist(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            self.make_run(folder)
            hbt.save_json(folder / "g2_plan.json", self.settings(delay_ps=2000, half_bins=4))
            sentinel = folder / "nist_analysis_existing"
            sentinel.mkdir()
            (sentinel / "interval_bits_A.bin").write_bytes(b"\xab\xcd")
            before = {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}
            with patch.object(hbt, "SHOW_FIGURES", False), contextlib.redirect_stdout(io.StringIO()):
                output = hbt.analyze_g2(folder)
            report = json.loads((output / "g2_report.json").read_text())
            self.assertEqual(report["status"], "estimated")
            self.assertEqual(report["settings"]["delay_ps"], 2000)
            self.assertGreater((output / "g2_correlation.pdf").stat().st_size, 1000)
            self.assertTrue((output / "g2_histogram.csv").exists())
            with np.load(output / "g2_histogram.npz", allow_pickle=False) as data:
                self.assertEqual(data["coincidences"].dtype, np.dtype("int64"))
            for path, contents in before.items():
                self.assertEqual(path.read_bytes(), contents)
            with patch.object(hbt, "SHOW_FIGURES", False), contextlib.redirect_stdout(io.StringIO()):
                reanalyzed = hbt.analyze_g2(folder, use_saved_settings=False)
            new_report = json.loads((reanalyzed / "g2_report.json").read_text())
            self.assertEqual(new_report["settings"]["delay_ps"], 0)
            self.assertNotEqual(output, reanalyzed)

    def test_data_loss_blocks_exports(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            info = self.make_run(folder)
            info["hardware_overflows"] = 1
            hbt.save_json(folder / "acquisition.json", info)
            with contextlib.redirect_stdout(io.StringIO()):
                output = hbt.analyze_g2(folder)
            report = json.loads((output / "g2_report.json").read_text())
            self.assertEqual(report["status"], "blocked")
            self.assertFalse((output / "g2_histogram.csv").exists())


if __name__ == "__main__":
    unittest.main()
