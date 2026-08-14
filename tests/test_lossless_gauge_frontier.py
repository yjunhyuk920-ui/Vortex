from scripts.derive_lossless_gauge_frontier import build_summary


def test_lossless_and_attention_ceilings() -> None:
    summary = build_summary()
    lossless = summary["lossless_streaming"]
    fast_link = lossless["fast_link"]
    attention = summary["attention_reparameterization"]

    assert lossless["shannon_bf16_bits_per_parameter"] == 10.6
    assert fast_link["minimum_zero_compute_tokens_for_20ms"] == 841
    assert fast_link["zero_compute_ms_per_token_at_32_tokens"] == 525.1467264
    assert attention["all_attention_parameters"] == 71_873_593_344
    assert attention["remaining_fraction_after_free_attention_deletion"] > 0.82
    assert "NO_SURVIVING_CANDIDATE" in summary["decisions"]
