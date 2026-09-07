# Stage 2: compositional proof, not increased solver budget
Stage 1's two UNKNOWN miters remain unknown; no solver timeout is promoted. Preserve
all first-run SMT/results. Follow-on frozen before execution:
- Recognize c=s*2^k, s odd, a BF16 constant, k>=-16, |c|<=1.
  For every finite BF16 x, cx is EXACT and finite in FP32. Proof: x is an integer
  multiple of 2^-133, product quantum >=2^-149, <=16 significant binary digits,
  magnitude <=max_finite_BF16 < max_finite_FP32.
- In FP32 add(acc,mul(c,x)), replace with fma(c,x,acc) only if that product is exact.
  No BF16 cast is crossed. Preserve the addition order and every output/state root.
- Shared-input add(mul(a,x),mul(b,x)) becomes mul(a+b,x) only for same-sign nonzero
  a,b, exactly representable BF16 a+b, and exact-product guards for all three.
- Explicit BF16-store witness x_bits=3 for a=3/8,b=1/8 will be checked; this is a
  proposed derived witness, not an assumed result.
- Three synthetic dense matrices m=4, n=8/16/32, seeds=81/161/321. Coefficients are
  BF16 with exponent field121..125 and arbitrary mantissa/sign. 128 finite BF16
  input vectors each. Also mixed extreme-weight matrix, seed999, m2,n4,64 inputs,
  with native all finite BF16 constants; unsafe products retain original operations.
- Compare C original, C transformed, independent exact Fraction FP32/BF16 emulator.
- Existing 8192-node C interpreter cap, input<=128, outputs<=128; not target engine.
- No >=10x claim: FMA is one instruction but still a multiplication and addition.
  Literal/code reads remain linear in the weights. Current GPUs already use FMA
  in many kernels; no baseline hardware or latency evidence is inferred.

Additional verification: the same frozen input vectors are checked at both FP32
intermediate outputs and final BF16 outputs; this strengthens checks without changing
seeds, candidate selection or cost thresholds.
