"""Offline regressions: python -m unittest -v test_hbt_nist_acquisition.py"""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import hbt_nist_acquisition as hbt


class NistWorkflowTests(unittest.TestCase):
    def test_published_examples(self):
        hbt.nist_selftest()
        hbt.selftest()

    def test_interval_pairing_and_packing(self):
        bits, counts = hbt.interval_comparison_bits(np.array([5, 3, 2, 4, 8, 8, 9]))
        np.testing.assert_array_equal(bits, [1, 0])
        self.assertEqual(counts, {"pairs": 3, "ties_discarded": 1, "unpaired_intervals": 1})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bits.bin"
            exported = hbt.write_bits(path, np.array([1, 0, 1, 0, 0, 1, 0, 1, 1], dtype=np.uint8))
            self.assertEqual(exported, 8)
            self.assertEqual(path.read_bytes(), b"\xa5")

    def test_timestamp_validation(self):
        for invalid in ([1., 2.], [[1, 2]], [1, 1], [2, 1], [-1, 2],
                        np.array([0, 2**63], dtype=np.uint64)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                hbt.timing_intervals(invalid)
        np.testing.assert_array_equal(hbt.timing_intervals(np.array([], dtype=np.int64)), [])
        np.testing.assert_array_equal(hbt.timing_intervals([0, 2**63 - 1]), [2**63 - 1])

    def test_fixed_sampling_rules(self):
        plan = hbt.sampling_plan("test")
        for key, value in (("bin_width_ps", 0), ("sample_count", True),
                           ("bits_per_symbol", 9), ("sp22_sequence_bits", 50),
                           ("sp22_block_bits", 128), ("sp22_alpha", float("nan"))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                hbt.validate_plan({**plan, key: value})

    def test_short_data_never_passes(self):
        plan = hbt.sampling_plan("test")
        report = hbt.sp22_assessment(np.zeros(99, dtype=np.uint8), plan)
        self.assertEqual(report["status"], "insufficient_bits")
        self.assertIsNone(report["summary"]["runs"]["uniformity_p"])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "raw.u8"
            path.write_bytes(bytes(100))
            with patch.object(hbt.subprocess, "run") as runner:
                report = hbt.run_non_iid(path, 8, "missing.exe")
            runner.assert_not_called()
            self.assertEqual(report["status"], "insufficient_samples")

    def test_biased_and_alternating_bits(self):
        plan = {**hbt.sampling_plan("test"), "sp22_sequence_bits": 100, "sp22_block_bits": 20}
        all_zero = hbt.sp22_assessment(np.zeros(100, dtype=np.uint8), plan)
        self.assertEqual(all_zero["runs_prerequisite_failures"], 1)
        self.assertEqual(all_zero["summary"]["runs"]["passes"], 0)
        alternate = hbt.sp22_assessment(np.tile([0, 1], 50), plan)
        self.assertEqual(alternate["per_sequence"][0]["frequency"], 1.)
        self.assertFalse(alternate["per_sequence"][0]["passes"]["runs"])

    def test_uniformity_requires_55_sequences_and_records_tails(self):
        plan = {**hbt.sampling_plan("test"), "sp22_sequence_bits": 100, "sp22_block_bits": 20}
        bits = np.random.default_rng(2026).integers(0, 2, 5503, dtype=np.uint8)
        short = hbt.sp22_assessment(bits[:5400], plan)
        full = hbt.sp22_assessment(bits, plan)
        self.assertIsNone(short["summary"]["frequency"]["uniformity_p"])
        self.assertEqual(full["summary"]["frequency"]["aggregate_status"], "evaluated")
        self.assertEqual(full["unused_tail_bits"], 3)
        json.dumps(full, allow_nan=False)

    def test_estimator_summary_parser_fails_closed(self):
        self.assertEqual(hbt.parse_non_iid_output("H_original: 0.75\n", 1)["initial_estimate_bits_per_symbol"], .75)
        for output in ("Most Common Value Estimate = 7.0", "H_original: nan",
                       "H_original: 9.0", "H_original: 2\nH_bitstring: 1.1"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                hbt.parse_non_iid_output(output, 8)
        with self.assertRaises(ValueError):
            hbt.parse_non_iid_output("H_original: 7.0\n", 8, require_bitstring=True)

    def test_estimator_execution_and_failure_statuses(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "raw samples.u8"
            np.resize(np.arange(256, dtype=np.uint8), 1_000_000).tofile(path)
            executable = Path(d) / "dummy.exe"
            executable.write_bytes(b"test fixture only")
            self.assertEqual(hbt.run_non_iid(path, 8)["status"], "not_run")
            with patch.object(hbt.shutil, "which", return_value=None):
                self.assertEqual(hbt.run_non_iid(path, 8, "missing")["status"], "tool_unavailable")
            def success(argv, **kwargs):
                self.assertFalse(kwargs["shell"])
                self.assertEqual(argv[1:], ["-i", "-a", "-v", path.name, "8"])
                self.assertEqual(kwargs["cwd"], path.parent.resolve())
                kwargs["stdout"].write("H_original: 6.0\nH_bitstring: 0.5\n")
                return subprocess.CompletedProcess(argv, 0)
            with patch.object(hbt.shutil, "which", return_value=str(executable)), \
                 patch.object(hbt.subprocess, "run", side_effect=success):
                report = hbt.run_non_iid(path, 8, str(executable))
            self.assertEqual(report["initial_estimate_bits_per_symbol"], 4.)
            self.assertEqual(report["status"], "completed")
            for effect, expected in ((subprocess.TimeoutExpired("dummy", 1), "timeout"),
                                     (OSError("unavailable"), "tool_error")):
                with patch.object(hbt.shutil, "which", return_value=str(executable)), \
                     patch.object(hbt.subprocess, "run", side_effect=effect):
                    self.assertEqual(hbt.run_non_iid(path, 8, str(executable))["status"], expected)
            with patch.object(hbt.shutil, "which", return_value=str(executable)), \
                 patch.object(hbt.subprocess, "run", return_value=subprocess.CompletedProcess([], 2)):
                self.assertEqual(hbt.run_non_iid(path, 8, str(executable))["status"], "tool_error")

    def test_raw_event_journal_preserves_error_tags(self):
        block = Mock()
        block.getTimestamps.return_value = np.array([1, 2, 3, 4])
        block.getChannels.return_value = np.array([2, 3, 0, 2])
        block.getEventTypes.return_value = np.array([0, 0, 2, 0])
        block.getMissedEvents.return_value = np.array([0, 0, 7, 0])
        stream = Mock()
        stream.getData.return_value = block
        buffers, counters = {2: [], 3: []}, {2: 0, 3: 0}
        with tempfile.TemporaryDirectory() as d:
            journal = [Path(d), 0]
            self.assertEqual(hbt.drain_stream(stream, buffers, counters, journal), (7, 1))
            with np.load(Path(d) / "block_00000000.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["event_types"], [0, 0, 2, 0])
                np.testing.assert_array_equal(saved["missed_events"], [0, 0, 7, 0])
            self.assertEqual(journal[1], 1)
            self.assertEqual(counters, {2: 2, 3: 1})

    def test_collect_manual_stop_and_hardware_overflow(self):
        for overflows, status in ((0, "stopped_by_user"), (1, "data_loss")):
            with self.subTest(overflows=overflows), tempfile.TemporaryDirectory() as d:
                block, empty = Mock(), Mock()
                for item, ts, channels in ((block, [1, 2, 3, 4], [2, 3, 2, 3]),
                                          (empty, [], [])):
                    item.getTimestamps.return_value = np.array(ts, dtype=np.int64)
                    item.getChannels.return_value = np.array(channels, dtype=np.int32)
                    item.getEventTypes.return_value = np.zeros(len(ts), dtype=np.int32)
                    item.getMissedEvents.return_value = np.zeros(len(ts), dtype=np.int64)
                driver = Mock()
                driver.createTimeTagger.return_value.getOverflows.return_value = overflows
                driver.TimeTagStream.return_value.getData.side_effect = [block, empty]
                with patch.object(hbt, "TimeTagger", driver), \
                     patch.object(hbt.time, "sleep", side_effect=[None, KeyboardInterrupt]):
                    raw, info = hbt.collect(folder=Path(d))
                np.testing.assert_array_equal(raw[2], [1, 3])
                self.assertEqual(info["status"], status)
                self.assertEqual(info["valid"], overflows == 0)
                self.assertEqual(info["raw_event_log_blocks"], 1)
                driver.TimeTagStream.return_value.stop.assert_called_once()
                driver.freeTimeTagger.assert_called_once()

    def test_plan_saved_before_hardware_and_raw_saved_before_analysis(self):
        with tempfile.TemporaryDirectory() as d:
            def collect(target, timeout, folder):
                plan = json.loads((folder / "sampling_plan.json").read_text())
                self.assertEqual(plan["provenance"], "before_acquisition")
                self.assertEqual(json.loads((folder / "g2_plan.json").read_text())["provenance"],
                                 "before_acquisition")
                return {2: np.array([1, 3]), 3: np.array([2, 4])}, {
                    "status": "complete", "error": None, "valid": True}
            def analyze(folder):
                self.assertTrue((folder / "raw_ch2.npy").exists())
                self.assertTrue((folder / "raw_ch3.npy").exists())
                self.assertTrue((folder / "acquisition.json").exists())
            with patch.object(hbt, "SAVE_ROOT", Path(d)), \
                 patch.object(hbt, "load_user_config"), \
                 patch.object(hbt, "configure_publication_plots"), \
                 patch.object(hbt, "collect", side_effect=collect), \
                 patch.object(hbt, "analyze", side_effect=analyze), \
                 patch.object(hbt, "analyze_g2"), \
                 patch.object(hbt, "characterize"), contextlib.redirect_stdout(io.StringIO()):
                hbt.run_acquisition(2, 1)

    def make_run(self, folder, n=401):
        intervals = np.resize(np.array([300000, 100000, 100000, 300000]), n - 1)
        timestamps = np.concatenate(([0], np.cumsum(intervals)))
        for c in (2, 3):
            np.save(folder / f"raw_ch{c}.npy", timestamps)
        info = {"valid": True, "status": "complete", "channels": [2, 3],
                "missed_events": 0, "non_photon_tags": 0, "hardware_overflows": 0,
                "raw_events": {"2": n, "3": n}, "target_A": n, "wall_seconds": 1.0}
        hbt.save_json(folder / "acquisition.json", info)
        return info

    def test_end_to_end_frozen_plan_and_loss_blocking(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            info = self.make_run(folder)
            plan = {**hbt.sampling_plan("before_acquisition"), "sample_count": 201,
                    "sp22_sequence_bits": 100, "sp22_block_bits": 20}
            hbt.save_json(folder / "sampling_plan.json", plan)
            before = hbt.sha256_file(folder / "raw_ch2.npy")
            with contextlib.redirect_stdout(io.StringIO()):
                output = hbt.analyze(folder)
            report = json.loads((output / "nist_report.json").read_text())
            self.assertEqual(report["status"], "analysis_complete")
            result = report["channels"]["A"]
            self.assertEqual(result["selected_intervals"], 201)
            self.assertEqual((output / "raw_timing_A.u8").stat().st_size, 201)
            self.assertEqual((output / "interval_bits_A.txt").read_bytes(), b"10" * 50)
            self.assertEqual(result["sp800_22"]["complete_sequences"], 1)
            self.assertEqual(result["sp800_90b"]["status"], "insufficient_samples")
            self.assertEqual(hbt.sha256_file(folder / "raw_ch2.npy"), before)
            info["missed_events"] = 1  # Even a stale valid=True flag cannot override loss.
            hbt.save_json(folder / "acquisition.json", info)
            with contextlib.redirect_stdout(io.StringIO()):
                blocked = hbt.analyze(folder)
            self.assertNotEqual(blocked, output)
            self.assertEqual(json.loads((blocked / "nist_report.json").read_text())["status"], "blocked")
            self.assertFalse((blocked / "interval_bits_A.txt").exists())

    def test_legacy_run_is_retrospective_and_malformed_input_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            self.make_run(folder)
            with contextlib.redirect_stdout(io.StringIO()):
                output = hbt.analyze(folder)
            report = json.loads((output / "nist_report.json").read_text())
            self.assertEqual(report["sampling_plan"]["provenance"], "retrospective_legacy_run")
            self.assertFalse((folder / "sampling_plan.json").exists())
            np.save(folder / "raw_ch3.npy", np.array([2, 1]))
            with contextlib.redirect_stdout(io.StringIO()):
                output = hbt.analyze(folder)
            self.assertEqual(json.loads((output / "nist_report.json").read_text())["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
