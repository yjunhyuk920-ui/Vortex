# Independent audit: paid Krylov feedback constructor

## Result

No correctness flaw found in the declared F2 recurrence or its n<=64 BF16 native realization. This is a finite, paid source-to-source temporal realization constructor, not a lookup table, state-enumeration procedure, or an HF executor. The current instrumentation is a partial cost inventory rather than a complete physical-resource upper bound.

The unmodified four-case suite was rerun into this directory. Thirty-two source/hot/decoder/input/CSV/candidate-trace files agree byte-for-byte with the supplied results. An independently formulated bit-by-bit transition check covered every A,b,c1,c2,co for n=1,2 and every represented state and input: 4,128 compiles, 20,576 transitions, all pass. Native extra tests cover r=64, including the top-bit update, and the maximum original accumulation 128, each for 256 steps. See independent_review.json and review_independent.py. These are scoped checks, not the proof or a complete repository/hardware test.

## Algebra and causal state

Let r be the first dependence index in b,Ab,..., so K=[b,...,A^(r-1)b] has independent columns and A^r b=Kf. The compiler's elimination labels maintain rem=v XOR K*coeff. Its stored pivot label represents the corresponding reduced pivot. Thus the first zero residual yields exactly f. Termination takes at most n independent columns followed by one dependence.

For r>0, Jz=((z<<1)&((1<<r)-1)) XOR f*z_(r-1), and b=Ke0. Columnwise AK=KJ. Transported d_i=c_i K gives the same two scalar gate inputs. Therefore s=Kz implies

s' = K[Jz XOR e0*(a XOR ((d1*z)&(d2*z)))].

The compiled output uses co*K against the new z, exactly matching the original observer on new s. Both execute the same uint32 LCG once and the same strict threshold comparison. Equal observations/RNG imply equal next own-output input, so induction covers arbitrary externally supplied bits and own-output continuations.

The state is always in span(K). In fact every z in F2^r is reachable: use the free external bit to cancel the current nonlinear gate, then controllability of (J,e0) reaches any coordinate in r steps. No state-size reduction is promised when r=n. For b=0, r=0, s remains zero and g=y=0, while the sampler still advances once. For rank-deficient nonzero b the same proof applies. The additional r=64 native test closes an absent implementation edge case without changing the theorem.

The recurrence-coefficient corruption by e0 is guaranteed to produce a state witness for r>0 under initial impulse: the highest coordinate is first one after r shifts, pre-error trajectories coincide, and the modified update differs by e0, decoding to nonzero b. The provided witnesses at steps 29,63,1 agree with this reasoning.

## Native semantics and source boundary

The native rows use BF16 values 1 or 2 and state values 0 or 1. Every product is 0,1,2; every nonnegative partial FP32 sum is an integer <=2n<=128; all are also exactly representable in BF16. Subtracting popcount(s) recovers the binary dot's integer count and parity recovers the F2 result. Thus no reassociation or finite-word subtlety affects this declared domain. The source compiler checks all coefficient words and charges a source scan. No source-dependent address/code is hidden in a free oracle.

This source program already contains integer subtraction and parity; it is not ordinary rounded HF MatVec. The BF16 257-to-256 witness correctly blocks unrestricted field substitution. No claim about all rounded programs follows from exactness on these bounded integer sums.

The supplied C program prints full state, inputs, gate and RNG for checking. Those are validation instrumentation under the plan's observer-only hot ABI. If the CSV is instead declared the actual external observation interface, Kz must be produced and charged every step. The runtime Executor.step itself does not access K or original A. Its validation wrapper does, and the summary separately discloses r basis-word reads per readback.

## Cost qualifications required

1. hot_bytes=52 counts serialized static n,r,f,d1,d2,do,mask and magic only. It excludes at least mutable z and RNG (12 bytes in a packed implementation), live input/output temporaries and buffers, executable/allocator overhead, and Python objects. Decoder bytes are 12+8r. Source bytes are 8+2(n^2+4n). Retained original source storage remains paid.
2. Postcheck matrix-row reads are counted, but the corresponding arithmetic, r^2 decoder loop visits/bit tests/basis accesses, three observer rechecks, and assertion/comparison overhead are not completely counted. Certification cannot be called fully paid merely from postcheck_matrix_row_reads.
3. Packed rows, source-word tuple/blob, pivot dictionaries and labels, columns, serialization temporaries and their peak overlap are missing from a complete memory inventory. Per-step gate/update/observer/RNG/address/output arithmetic is not enumerated. Source/header parsing/validation costs also exceed the five-operation row bundles.
4. A finite polynomial bound is easy at n<=64: construction and certification require O(n^2+nr+r^2) bounded-word operations and O(n+r) packed payload words plus the original/serialization buffers, with implementation-specific Python overhead extra. This does not substitute for a measured byte peak or latency bound.
5. The source-isolation claim is sound: fixed Executor code plus the hot bytes determine all observer-only continuations. K is a separately retained exact decoder. For generic r=n, calling the literal decoder every step reads n packed columns and performs up to n XORs, restoring an O(n^2)-bit basis access pattern. Narrow observations are essential to the claimed hot-source reduction.
6. The gate-removal ablation holds the unablated candidate's inputs fixed in both modes. In own-output mode this is a same-input causal ablation, not comparison of two independently sampled feedback loops.
7. No timing/break-even value follows from native product counts versus packed popcount counts. A break-even claim needs comparable operation/service costs for constructor, serialization, execution and readback, and a specified finite horizon.

## Counterexample boundaries

The result refutes the claim that a repeatedly used dense source must be independently queried at every step. It does not contradict the address-fiber theorem, whose Cartesian independent-query setting differs from a source-dependent reachable dynamical state with paid common advice.

The favorable algebra depends on one repeated linear A and rank-one nonlinear innovation through b, plus only three scalar projections. Multiple independently selected A matrices generally become dense operators in one common Krylov basis. A number of feedback/output projections proportional to n requires O(nr) stored projection bits and corresponding online projection work. Arbitrary initial state needs additional paid encoding/a larger invariant module; s=0 is essential to this exact stated constructor. Whole-state readback and source-copy retention cannot be silently discarded.

## Most precise next native relation worth trying

A concrete extension would need source-constructed native-bit state embeddings K_t (with an affine constant coordinate if needed), cheap transport J_t, and narrow innovation maps B_t, for the actual token/layer boundary native program F_(W,t), such that on every legally reachable encoded state:

bits(F_(W,t)(decode_bits(K_t z), a, rho))
  = K_(t+1) [J_t z XOR B_t eta_(W,t)(z,a,rho)].

All required live state, KV/layout/control and RNG successor bits must be included or handled by an explicit exact parallel recurrence. The original exposed outputs must also admit a source-constructed cheap observer on the encoded transition. The decisive new content is a causal algorithm for eta and the observer BEFORE evaluating the eliminated native body, including exact reduction order, rounding, carries, nonlinearities and attention dependencies. Their total width, source probes, computation and maintenance must fit a sufficient bound, as must construction/storage of K_t,J_t,B_t across the supported context range.

Defining eta as the difference after running the original region, choosing a full-width B=I, tabulating all reachable states, or merely asserting the displayed equality supplies no useful extension. A field-exact guard on a tiny subdomain and the prior inverse-paired conjugacy do not construct this native innovation source. This is a falsifiable missing relation and cost obligation, not an admitted solution; no target model/backend run is justified by this review alone.

No root ledgers, publication state or repository science files were edited. Parent owns remote/provenance integration.
