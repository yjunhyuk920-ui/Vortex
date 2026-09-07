# Pre-execution plan — selector adjoint source, 2026-09-07
Base: Vortex PR141, 8cb10dab1d3d1fdc2e2dc857b5db5bb9500def09.
Mission unchanged: universal unmodified 405B HF dense, same output/RNG/successor state, one <=8GiB GPU, 4B Q4 p50/p95/TTFT and all costs paid.

The final mission theorem is NOT established. O1 constructive cheap source OPEN; O2 native full-program equivalence OPEN; O3 causal full-state bridge OPEN; O4 complete storage/workspace OPEN; O5 paid build/query latency OPEN; O6 reproducible target hardware validation NOT TESTED.

Three candidate principles (proposals, not three qualifying >=10x inventions):
A. Recover all required output/state bits by transposing one selector-linear Boolean scalar program. Prove a typed linear-extraction algorithm; do not differentiate rounded real arithmetic as though it were associative. The free scalar circuit is the missing source obligation, not an oracle we may assume.
B. Replace forward values by the unique ground state of local gate-constraint energy. Test whether uniqueness gives cheap/greedy construction; count the source and solver, not just the final scalar energy.
C. Reconstruct all coordinates from expectations of directional scalar evaluations. Derive the rank requirement of this particular fixed linear reconstruction and preserve random-error witnesses. This is not a lower bound on nonlinear decoders or all executors.

Cost-first admission: require >=10x reduction of the dominant full-source work AND a plausible storage/build/state/native route before a neural backend or larger tuning. If no route passes, implement only the small algebra/counterexample checker, preserve the gate, and do not label an engine as completed.

Fixed checks (before observing results):
1. FP32-bit to BF16 RNE, canonical positive NaN 0x7fc0, signed zeros/overflow/infinities included. All 65536 upper halfwords times low halfwords [0,32767,32768,32769,65535], plus 4096 RNG seed17029 raw words. The successor-state roots copy the full original 32-bit input. This is a conversion primitive, not a Transformer. Independent C frexp/ldexp reference and numpy-independent Fraction spot checks.
2. Arbitrary dense GF(2) matrices n=[16,32,64,128], seeds=[17,29], 256 binary vectors each. Encode transpose-first selector sources; compare same GF(2) bit-packed baseline, not BF16 baseline. GF(2) parities are NOT numeric sums.
3. 12 random Boolean DAGs, input width8, all256 inputs, 16 root bits. Verify extraction against direct evaluation and single-selector probes. Same engine supplied with serialized data only.
4. Native rounding-order witness fixed weights [2^24,1,-2^24,1], all inputs1; balanced reduction vs backward sequential accumulation. Native-state liveness is tested separately from current-output equality.
5. Energy copy chain with one branching node; all8 assignments. Directional +/-1 n=2 exhaustive and exact frame n=4; finite linear-frame rank test. No stochastic output accepted as deterministic exactness.

Report measured logical gate/file counts separately from hardware traffic or latency. No real pretrained checkpoint, CUDA, 405B, full KV/RNG, VRAM or 4B benchmark is promised. All failures, sources, input/output hashes and replays will be saved. No large truth table, full neural backend, automatic higher-dimension tuning, or retesting favorable matrix families.
