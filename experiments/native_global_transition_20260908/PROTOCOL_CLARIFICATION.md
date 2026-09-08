# Mask/padding protocol clarification — same frozen population

After the guarded run_v2, the user's queued instruction requested resolution of
the HF warnings. run_v2 and first failed run remain intact. run_v3/replay use
the SAME three token-ID prefixes, seeds and sample settings, explicitly passing
an all-one int64 attention mask and pad_token_id=0, equal to the pinned HF
generation code's previously warned auto pad=eos0. EOS itself is not changed.
No prompt, seed, temperature, candidate or threshold is selected after results.

This is an ABI ambiguity check, not an attempt to rescue a >=10x claim. Exact
reference/candidate logits, every KV tensor bit and shape/stride/offset, each
forward's before/after RNG state and final RNG state are compared. A separately
saved comparison checks run_v2 reference outputs and sampled sequences against
run_v3 reference. A/B/C retain every dominant native matrix operation.

Unsupported: padded/custom masks, sliding window, other attention backends,
batch>1, other model classes/checkpoints, model/config mutation, other devices,
CUDA, the complete legal-context population. The wrapper doesn't establish the
whole mission's O3. Native kernels, the original checkpoint and decoder/codec
costs are not free, and no original output is provided to a candidate.
