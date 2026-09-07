# Token-addressed causal cut: bounded construction and coverage gate

2026-09-07; parent PR133 `7cf9c4ac579060ecfbb0dd91a0ba2bda27d2b490`.
Fixed mission and CTC-2026-09-05 are unchanged. This is a small semantic/cost probe,
not a new admitted core experiment or a completed 405B executor.

## Concrete method and proof scope
For token ID t, compile T[t]=F_W(E[t])=(q0,k0,v0) before first-layer RoPE.
Run the same deterministic single-token ABI for every vocabulary entry; runtime
pread address is header+2(h+2k)t. The query object has no original Q/K/V/norm arrays.
All remaining RoPE, attention, layers, logits, KV and sampler operations are unchanged.
If each entry equals the native prefix and it depends only on t, equal cut bits plus
equal prior state imply equal next state inductively. Arbitrary weight structure is
allowed at this particular cut, not arbitrary architectures or all downstream cuts.
Batched construction is not presumed numerically equivalent or free.

The implemented ABI is a synthetic NumPy CPU decoder with BF16 stores, separate FP32
products and balanced FP32 reductions. It is NOT the HF/CUDA/Tensor Core ABI. Direct
inputs_embeds, changed adapters, position/history before the cut and dynamic numerical
ABIs need additional treatment. No universal original-checkpoint proof is claimed.

## Results and decisive scope boundary
Two three-layer synthetic dense checkpoints, seeds7/29, V17/31, h16/32:
48 vocabulary IDs and2528 cut coordinates match both direct and independent scalar
projection results. Eight causal runs total160 steps/128 generated tokens; logits,
KV and RNG all match. The table runner has first-cut arrays removed. No target token
oracle, input approximation or fallback is used. Original checkpoint bytes persist.

Prefixes[1,3] and[2,3] give the same first cut but different second-layer q/k/v at
current token3,position1. Seed7 second query component is1.703125 vs-0.609375.
Thus token+position alone cannot extend this table to that second-layer value.
This is not a lower bound against all context compression or other algorithms.

## Whole-cost gate (DERIVED, not target measurements)
Official Meta405B bf16-mp8 example: h16384,k1024,f53248,L126,V128256.
A=h(h+2k)=301989888; P=L[A+h^2+3hf]+Vh=403747897344 projection MAC/token.
Only A/P=0.0747966466% is removed;99.9252033534% remains. This is a projection
work ratio, not latency, physical memory traffic or an attention-inclusive model.
The table payload is4728029184 bytes=4.4033203125GiB, row36864 bytes=36KiB.
Other weights, KV, buffers and metadata remain. Single-token construction uses
38732015075328 MAC plus norm/storage/hash work; logical weight operand references
are77464030150656 bytes, NOT measured SSD/PCIe traffic. All payload is verified
on load and charged. Prototype holds the whole small model and copies temporary
arrays; no target streaming constructor or peak8GiB bound was implemented.

The other compared principles are exhaustive next-token states and deferred KV
recipes. Direct enumeration produces61.646484375GiB of new KV payload and V*P work,
only one branch consumed. Lazy storage alone does not remove current-step KV use.
Known tabulation/branching/laziness are not three new qualifying inventions.
All three fail core admission; no larger table, branch backend or GPU port was built.

## Status and ledger addendum
THEORY_STATUS=NOT_ESTABLISHED; HARDWARE_STATUS=NOT_TESTED; CORE_ADMISSION=false;
THREE_QUALIFYING_NEW_PRINCIPLES=false; full-mission O1-O6=OPEN.
O2/O3 receive a bounded cut replacement lemma, not a universal executor. O4 receives
explicit partial costs and an admission rejection, not a sufficient mission bound.
No public pretrained checkpoint, target GPU/405B, native4BQ4/TTFT, full repository
suite or Actions run. Tests and remote persistence do not establish scientific success.
Next source must handle actual history-dependent values cheaply; extending this
lookup without that construction simply moves the same hard work. Preserve prior
failures, ABI limits and complete preparation/state accounting. Root historical
ledgers/policies are unchanged; this is the explicitly linked current addendum.

## Reproduction and complete bounded evidence
The four capsule parts concatenate to the SHA in capsule_manifest.json. Restore17
text files including all source/tests, preregistration, full Korean report, recorded
summary, costs, context witnesses and original32-file hash manifest. Binary inputs,
source tables, final states and detailed per-step traces are regenerated against
that manifest; they are not all embedded in the remote text capsule. Full originals
are supplied in the user ZIP. Archival XZ is not an inference compression result.

```bash
python experiments/causal_cut_20260907/restore.py --out /tmp/vortex-causal-cut
cd /tmp/vortex-causal-cut
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python src/analyze.py --out regenerated
python src/check_reproduction.py regenerated --manifest results/manifest.json
```

20 tests passed and32 generated files reproduced byte-exactly, including an
independent restore/re-execution from the text capsule. Read docs/REPORT_KO.md in
the restored directory for equations, assumptions, O1-O6 ledger and primary sources.
Prior local rounding_monoid/carry_source_gate archives are NOT retroactively included.
Remote ref/PR read-back is required before reporting REMOTE_COMMIT_VERIFIED.
