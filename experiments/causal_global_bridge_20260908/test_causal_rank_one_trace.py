from __future__ import annotations

import unittest

from causal_rank_one_trace import RankOneTraceConfig, trace_audit


class CausalRankOneTraceTests(unittest.TestCase):
    def test_32_query_right_factor_trace_survives_native_finite_words(self) -> None:
        payload = trace_audit(
            RankOneTraceConfig(right_bits=8, output_rows=4, queries=32, seed=21)
        )
        self.assertEqual(payload["sequence_length"], 8 + 1 + 31 * (16 + 1))
        self.assertEqual(payload["query_hidden_sign_or_common_magnitude_failures"], 0)
        self.assertEqual(payload["full_vs_incremental_query_logit_mismatches"], 0)
        self.assertEqual(payload["row_parity_decode_mismatches"], 0)
        self.assertEqual(payload["random_left_rank_one_scalar_decode_mismatches"], 0)
        self.assertEqual(payload["same_parity_bf16_codebook_collisions"], 0)
        self.assertEqual(payload["exhaustive_binary_rows_checked"], 32 * 256)
        self.assertEqual(payload["exhaustive_binary_row_decode_mismatches"], 0)
        self.assertTrue(payload["rng_unchanged_after_full"])
        self.assertTrue(payload["rng_unchanged_after_incremental"])

    def test_wider_two_query_trace(self) -> None:
        payload = trace_audit(
            RankOneTraceConfig(right_bits=108, output_rows=25, queries=2, seed=22)
        )
        self.assertEqual(payload["query_hidden_sign_or_common_magnitude_failures"], 0)
        self.assertEqual(payload["full_vs_incremental_query_logit_mismatches"], 0)
        self.assertEqual(payload["row_parity_decode_mismatches"], 0)
        self.assertEqual(payload["random_left_rank_one_scalar_decode_mismatches"], 0)
        self.assertEqual(payload["same_parity_bf16_codebook_collisions"], 0)


if __name__ == "__main__":
    unittest.main()
