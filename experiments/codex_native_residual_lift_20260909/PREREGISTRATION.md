# Two adaptive native BF16 count queries — preregistration

This is an O3 auxiliary reduction on the existing native arithmetic bridge,
not a new core direction or one of the missing three execution principles.
It does not construct the arbitrary-native fast decoder. Rejected tables,
full scans, free witnesses and related-input replay do not become fast cores.

Intended lemma: for every binary B[m,n], binary v[n], 1<=n<=16384, and finite
positive normal BF16 alpha0/alpha1 with unbiased exponent in [-100,100], compile
one fixed BF16 C=[B | row-private signed power-of-two bias lanes].
Let b=ceil(log2(n+1)); D=n+m*b. C[i,n+i*b+j]=-2^j, all other bias entries zero.
Only inputs change between calls; C is fixed before either result is observed.

Declared native ABI: separate FP32 products plus any binary FP32 sum tree,
or FP32 FMA accumulation, without reduced-precision partial sums, then RNE
BF16 store. No overflow/underflow occurs in the admitted range. This is not
an assertion about every CUDA/native kernel or arbitrary mixed-precision ABI.

Call1 x0=alpha0*(v,zeros); observe actual native y0.
For each row set k0=clamp(RNE exact-rational y0/alpha0,0,n).
Call2 x1=alpha1*(v,row-private binary digits of k0); observe native y1.
Decode k=k0+RNE exact-rational y1/alpha1, then parity=k mod2.
True Bv is never used to form either input or bias, only to verify afterward.

Proof obligations: exact FP32 intermediate bound for every native summation
order; BF16 first-error bound; clamp cannot increase error; second residual
rounding error below half an integer; finite integer implementation of ratio
RNE; zero signs; all compiler/query/storage/control costs. Intended bound:
|k-k0|<=65 at n<=16384, then residual BF16 error/alpha1<=65/256<1/2.

Cheapest decisive controls: a single-query same-BF16-output/different-count
collision; two-query recovery for zeros, all-ones, high counts, heterogeneous
rows and scales including exponent endpoints; n8192 and n16384. CPU only.
Primary independently runs a native PyTorch BF16 projection control after
the ordered FP32/NumPy implementation. Actual HF legal-token causality, full
RNG/KV/all-continuation relation are separate OPEN obligations; arbitrary
projection vectors are not automatically legal hidden states.

The scale audit uses the registered 405B projection shapes to count row-private
bias overhead and source freedom. The procedure executes TWO full augmented
matvecs and is not a >=10x path. No source-scale storage/read is hidden.
The theorem direction is exact counts/parity REDUCED TO native queries, not
native effects produced from a cheap parity oracle. No direct-sum allocation
of globally mixed nonlinear advice and no target lower-bound claim.

Primary owns proof/cost/judgment/integration. Terra/medium requested implementation;
Sol/high requested narrow independent proof review; actual model metadata not
exposed. No recursive delegation, GPU, agent Git or remote computation.
User has no405B hardware. Fixed universal mission remains unchanged:
O1-O5 OPEN/O6 PARTIAL, theory NOT_ESTABLISHED/hardware NOT_TESTED/corefalse.

