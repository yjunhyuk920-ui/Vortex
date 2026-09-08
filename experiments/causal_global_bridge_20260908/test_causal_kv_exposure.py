from __future__ import annotations

import unittest

import torch

from causal_kv_exposure import (
    ExposureConfig,
    build_exposure_model,
    cache_layer_values,
    caches_bit_equal,
    checkpoint_sha256,
    derive_exposure_audit,
    make_binary_matrices,
    run_full_prefill,
    run_incremental,
    validate_exposure_config,
    verify_cache_exposure,
)


class CausalKVExposureTests(unittest.TestCase):
    def test_constructor_uses_all_legal_tokens_as_basis(self) -> None:
        config = ExposureConfig(hidden_size=8, layers=2, seed=1)
        matrices = make_binary_matrices(config)
        model = build_exposure_model(config, matrices)
        embedding = model.model.embed_tokens.weight.detach().cpu().float()
        self.assertTrue(torch.equal(embedding, torch.eye(8)))

    def test_full_prefill_reconstructs_every_source_bit(self) -> None:
        config = ExposureConfig(hidden_size=8, layers=2, seed=2)
        matrices = make_binary_matrices(config)
        model = build_exposure_model(config, matrices)
        tokens = list(range(8))
        output = run_full_prefill(model, tokens)
        exposure = verify_cache_exposure(output.past_key_values, matrices, tokens)
        self.assertEqual(exposure["coordinates_checked"], 128)
        self.assertEqual(exposure["mismatches"], 0)
        self.assertEqual(len(exposure["nonzero_bf16_bit_patterns"]), 1)

    def test_arbitrary_legal_token_sequence_selects_columns(self) -> None:
        config = ExposureConfig(hidden_size=8, layers=3, seed=3)
        matrices = make_binary_matrices(config)
        model = build_exposure_model(config, matrices)
        tokens = [7, 0, 7, 3, 1, 6]
        output = run_full_prefill(model, tokens)
        exposure = verify_cache_exposure(output.past_key_values, matrices, tokens)
        self.assertEqual(exposure["coordinates_checked"], 3 * 6 * 8)
        self.assertEqual(exposure["mismatches"], 0)

    def test_incremental_continuation_matches_full_prefill_cache_bits(self) -> None:
        config = ExposureConfig(hidden_size=8, layers=2, seed=4)
        matrices = make_binary_matrices(config)
        model = build_exposure_model(config, matrices)
        tokens = [0, 5, 1, 7, 3, 2, 6, 4]
        full = run_full_prefill(model, tokens)
        incremental_cache, _ = run_incremental(model, tokens)
        self.assertTrue(
            caches_bit_equal(full.past_key_values, incremental_cache, config.layers)
        )

    def test_forward_does_not_mutate_checkpoint(self) -> None:
        config = ExposureConfig(hidden_size=8, layers=2, seed=5)
        matrices = make_binary_matrices(config)
        model = build_exposure_model(config, matrices)
        before = checkpoint_sha256(model)
        run_full_prefill(model, list(range(8)))
        after = checkpoint_sha256(model)
        self.assertEqual(before, after)

    def test_registered_audit_keeps_core_claims_open(self) -> None:
        payload = derive_exposure_audit(
            ExposureConfig(hidden_size=8, layers=2, seed=6)
        )
        self.assertEqual(payload["exact_equation"]["mismatches"], 0)
        self.assertTrue(
            payload["causal_state"]["full_prefill_incremental_cache_bit_equal"]
        )
        self.assertTrue(payload["rng"]["unchanged_after_full_prefill"])
        self.assertTrue(payload["rng"]["unchanged_after_incremental"])
        self.assertFalse(
            payload["claim_boundary"]["arbitrary_checkpoint_accelerator_constructed"]
        )
        self.assertFalse(payload["claim_boundary"]["target_405b_executed"])
        self.assertEqual(payload["obligation_update"]["O1"], "OPEN")
        self.assertEqual(payload["obligation_update"]["O5"], "OPEN")

    def test_rectangular_gqa_32_by_224_is_legal_and_exact(self) -> None:
        config = ExposureConfig(
            hidden_size=224,
            value_size=32,
            head_dim=2,
            layers=1,
            seed=7,
        )
        value_size, head_dim, q_heads, kv_heads = validate_exposure_config(config)
        self.assertEqual((value_size, head_dim, q_heads, kv_heads), (32, 2, 112, 16))
        matrices = make_binary_matrices(config)
        model = build_exposure_model(config, matrices)
        self.assertEqual(tuple(model.model.layers[0].self_attn.v_proj.weight.shape), (32, 224))
        tokens = [0, 223, 17, 91, 17, 3]
        full = run_full_prefill(model, tokens)
        exposure = verify_cache_exposure(full.past_key_values, matrices, tokens)
        self.assertEqual(exposure["coordinates_checked"], len(tokens) * 32)
        self.assertEqual(exposure["mismatches"], 0)

    def test_nonintegral_gqa_replication_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_exposure_config(
                ExposureConfig(
                    hidden_size=225,
                    value_size=24,
                    head_dim=3,
                    layers=1,
                )
            )


if __name__ == "__main__":
    unittest.main()
