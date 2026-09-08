# Implementation corrections (not scientific threshold changes)

The first full-forward graph test preserved all61 roots and layouts; matrix MAC
counts stayed538472576 before/after. First generation attempt stopped in the
official HF mask builder at `padding_mask.all()` because make_fx forbids reading
a traced scalar. This is a capture limitation, not numerical mismatch or a proof
that global native elimination is impossible. The original error log, source
snapshot and partial results are preserved under logs/attempt01_source and
results/run. Subsequent results use a new directory, never overwrite that run.

Read the actual pinned `masking_utils.py` lines202-245,680-815. For CPU SDPA,
unprocessed DynamicCache, no sliding/custom mask, q_length1 OR empty-cache
prefill, a full-width all-one padding mask and None both yield the same None
causal mask. The wrapper now checks this concrete guard OUTSIDE tracing and
charges a mask scan. Only the guarded call uses None; the external original HF
generation keeps its mask, original positions and sampler. Other masks/branches
are rejected, not silently specialized. All declared prompts are unchanged.

This narrows the tested ABI and is recorded as such; it cannot prove every legal
original input. Native general padding/masking and other model classes remain
OPEN. C now has an actual TOP/singleton demand interpreter on the same complete
roots: no desired answer is supplied, so inverse(TOP)=TOP and paid functional
evaluation remains. It does not materialize2^w word sets or claim a smaller
domain is free. Native kernels are always counted.
