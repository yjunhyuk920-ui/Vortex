from __future__ import annotations

import unittest

import torch

from basis_column_producer import (
    compile_basis_column_store,
    decode_value_vector,
    public_transition_bit_equal,
    run_compiled_incremental,
)
from causal_kv_exposure import (
    ExposureConfig,
    make_binary_matrices,
    run_incremental,
)


class BasisColumnProducerTests(unittest.TestCase):
    def test_packed_column_decoder_matches_binary_source(self) -> None:
        config = ExposureConfig(hidden_size=8, layers=2, seed=11)
        matrices = make_binary_matrices(config)
        store, _ = compile_basis_column_store(config, matrices)
        for layer in range(config.layers):
            scale = decode_value_vector(store, layer, 0)[
                matrices[layer][:, 0].bool()
            ]
            self.assertTrue(bool(torch.all(scale != 0)))
            for token in range(config.hidden_size):
                decoded = decode_value_vector(store, layer, token)
                self.assertTrue(
                    torch.equal((decoded != 0).to(torch.uint8), matrices[layer][:, token])
                )

    def test_compiled_square_transition_is_bit_exact(self) -> None:
        config = ExposureConfig(hidden_size=8, layers=3, seed=12)
        matrices = make_binary_matrices(config)
        store, model = compile_basis_column_store(config, matrices)
        tokens = [7, 0, 3, 3, 5, 1, 6, 2]
        native_cache, native_logits = run_incremental(model, tokens)
        compiled_cache, compiled_logits = run_compiled_incremental(store, tokens)
        self.assertTrue(
            public_transition_bit_equal(
                native_cache,
                native_logits,
                compiled_cache,
                compiled_logits,
                config.layers,
            )
        )

    def test_compiled_rectangular_gqa_transition_is_bit_exact(self) -> None:
        config = ExposureConfig(
            hidden_size=224,
            value_size=32,
            head_dim=2,
            layers=2,
            seed=13,
        )
        matrices = make_binary_matrices(config)
        store, model = compile_basis_column_store(config, matrices)
        tokens = [223, 0, 17, 91, 17, 3, 222, 44]
        native_cache, native_logits = run_incremental(model, tokens)
        compiled_cache, compiled_logits = run_compiled_incremental(store, tokens)
        self.assertTrue(
            public_transition_bit_equal(
                native_cache,
                native_logits,
                compiled_cache,
                compiled_logits,
                config.layers,
            )
        )
        self.assertEqual(store.words_per_column, 1)
        self.assertEqual(store.runtime_payload_bits_per_token, 2 * 64)
        self.assertEqual(store.source_binary_bits, 2 * 32 * 224)

    def test_compiled_step_does_not_consume_torch_rng(self) -> None:
        config = ExposureConfig(hidden_size=8, layers=2, seed=14)
        matrices = make_binary_matrices(config)
        store, _ = compile_basis_column_store(config, matrices)
        before = torch.random.get_rng_state().clone()
        run_compiled_incremental(store, [0, 4, 7, 1])
        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()
