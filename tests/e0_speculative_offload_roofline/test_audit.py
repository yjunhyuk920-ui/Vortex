from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments/e0_speculative_offload_roofline/run_audit.py"
CONFIG = ROOT / "experiments/e0_speculative_offload_roofline/config.json"
COMMITTED = ROOT / "results/e0_speculative_offload_roofline/summary.json"

spec = importlib.util.spec_from_file_location("roofline_audit", SCRIPT)
assert spec and spec.loader
roofline_audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roofline_audit)


class DualRooflineAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = roofline_audit.load_config(CONFIG)
        self.summary = roofline_audit.compute_audit(self.config)

    def test_registered_target_constants(self) -> None:
        floor = self.summary["target_floor"]
        self.assertAlmostEqual(floor["raw_exact_weight_gib"], 755.953125, places=9)
        self.assertAlmostEqual(floor["compressed_exact_weight_gib"], 503.96875, places=9)
        self.assertAlmostEqual(
            floor["compressed_target_sweep_io_s"], 67.64136445386949, places=9
        )
        self.assertAlmostEqual(
            floor["dense_candidate_floor_s"], 0.1887670900688372, places=12
        )
        self.assertEqual(floor["crossover_perfect_chain_tokens"], 359)

    def test_tree_inflation_cannot_improve_dense_floor(self) -> None:
        rows = {row["id"]: row for row in self.summary["diagnostic_scenarios"]}
        perfect = rows["perfect_chain_128"]
        cats = rows["cats_vicuna7b_mtbench_greedy_favorable_minimum"]
        subspec = rows["subspec_qwen25_7b_mtbench_diagnostic"]
        self.assertEqual(perfect["verified_nodes_per_accepted_token"], 1.0)
        self.assertGreater(cats["verified_nodes_per_accepted_token"], 1.0)
        self.assertGreater(subspec["verified_nodes_per_accepted_token"], 10.0)
        self.assertGreater(
            subspec["per_accepted_token_floor_s"],
            perfect["per_accepted_token_floor_s"],
        )

    def test_sub_critical_4b_baseline_rejects_unreduced_dense_chain(self) -> None:
        rows = self.summary["conditional_native_4b_rows"]
        below = [row for row in rows if row["native_4b_p50_ms"] <= 100.0]
        self.assertTrue(below)
        self.assertTrue(
            all(not row["unreduced_dense_chain_semantically_feasible"] for row in below)
        )
        critical = self.summary["target_floor"][
            "critical_native_4b_p50_ms_for_unreduced_dense_chain"
        ]
        self.assertAlmostEqual(critical, 157.30590839069767, places=9)

    def test_fully_resident_substitute_floor_exceeds_hot_budget(self) -> None:
        draft = self.summary["draft_residency_floor"]
        self.assertAlmostEqual(
            draft["full_model_four_bit_substitute_gib"], 188.98828125, places=9
        )
        self.assertAlmostEqual(
            draft["full_model_one_bit_substitute_gib"], 47.2470703125, places=9
        )
        self.assertFalse(draft["full_model_one_bit_substitute_fits_8gib"])
        self.assertLess(
            draft["maximum_whole_model_bits_per_parameter_in_8gib"], 0.17
        )

    def test_more_compression_reduces_only_io_term(self) -> None:
        stronger = json.loads(json.dumps(self.config))
        stronger["mission"]["favorable_lossless_compression_ratio"] = 2.0
        result = roofline_audit.compute_audit(stronger)
        self.assertLess(
            result["target_floor"]["compressed_target_sweep_io_s"],
            self.summary["target_floor"]["compressed_target_sweep_io_s"],
        )
        self.assertEqual(
            result["target_floor"]["dense_candidate_floor_s"],
            self.summary["target_floor"]["dense_candidate_floor_s"],
        )

    def test_cli_matches_committed_summary_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.json"
            second = Path(tmp) / "second.json"
            for output in (first, second):
                subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        "--config",
                        str(CONFIG),
                        "--output",
                        str(output),
                    ],
                    check=True,
                    cwd=ROOT,
                )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first.read_bytes(), COMMITTED.read_bytes())


if __name__ == "__main__":
    unittest.main()
