# Exact Whole-SwiGLU Global Compiler Gate

Status before hosted execution: **PREREGISTERED CONSTRUCTIVE COMPILER-SUBSTRATE GATE**

This Gate implements the user's requested research order: build the global nonlinear exact compiler first, validate whether it discovers a target-scale execution law, and only then justify a broader executor.

It does not lower the fixed VORTEX mission and it is not itself a surviving core candidate merely because the compiler runs.

## 1. Fixed question

Can an unchanged public checkpoint's complete SwiGLU function be compiled and reasoned about as one proof-carrying nonlinear program instead of three independent matrix calls?

For one real MLP:

```text
g(x) = W_gate x
u(x) = W_up x
h(x) = SiLU(g(x)) * u(x)
F_W(x) = W_down h(x)
```

The compiler unit is the complete function `F_W`. Matrix-local representations remain permitted only as a safe initial lowering of this global semantic object. They are not treated as the discovered breakthrough.

## 2. Why this changes the research interface

The previous rounding/session-basis path failed because a small linear basis for `x` did not remain small after the elementwise SwiGLU nonlinearity. Optimizing `W_gate`, `W_up`, and `W_down` independently therefore left almost all `down_proj` work.

This Gate moves the compiler boundary outward:

```text
three unrelated MatVec objects
    -> one nonlinear fiber program
    -> exact checkpoint-generated rewrites
    -> existing-ISA lowering
```

Each intermediate fiber `i` is the checkpoint-derived triple

```text
(W_gate[i,:], W_up[i,:], W_down[:,i])
```

and contributes

```text
W_down[:,i] * SiLU(W_gate[i,:] x) * (W_up[i,:] x).
```

This permits future compiler rules to share, combine, cancel, or index complete nonlinear contributions. It does not assume such a rewrite exists.

## 3. Initial proof-carrying IR

The first executable IR contains:

1. joint gate/up fiber tiles;
2. the exact checkpoint activation function;
3. the exact elementwise product;
4. complete `down_proj` output-row tiles;
5. hashes for every cold payload and decoded tensor;
6. a frozen-native-ABI guard;
7. a structural whole-function audit.

The down-projection reduction axis is never split. Output-row tiling is allowed because it does not regroup one output dot product.

## 4. Native ABI and fail-closed rule

Stacking gate and up into one larger affine call can change a floating kernel's reduction schedule. It is therefore not assumed exact.

The compiler uses this rule:

```text
non-BF16 ABI
    -> joint payload materialization
    -> two reference-shaped affine calls

pinned BF16 ABI
    -> run frozen bitwise compile probes
    -> use one stacked call only when every probe matches
    -> otherwise use the proof-preserving two-call lowering
```

Even a compile-probe pass is not a universal theorem. Hosted validation must replay all frozen prompts for 128 consecutive transitions and compare both tokens and successor state. A mismatch rejects the selected lowering.

Artifact corruption, shape mismatch, unsupported `pretraining_tp`, missing metadata, or failed integrity checks aborts. There is no approximate accept path.

## 5. Automatic whole-function audit

The compiler records, without prompt selection:

- exact duplicate and sign-antipodal gate rows;
- exact duplicate and sign-antipodal up rows;
- exact duplicate gate/up functions;
- exact duplicate and sign-antipodal down columns;
- exact duplicate complete nonlinear fibers;
- an identity-CSE optimistic coefficient fraction;
- a bounded joint-fiber Q4 spanning-tree screen.

The Q4 screen concatenates each quantized gate row, up row, and down column into one nonlinear-fiber signature. Collision-free 32-coefficient block IDs produce a favorable lower bound on every Hamming-difference tree over those fibers.

That screen is exact only for its symmetric-row-Q4 integer surrogate. It cannot establish BF16 semantics or wall-clock speed. It is included because it can cheaply kill one obvious global-fiber rewrite before synthesis work.

## 6. Fixed public checkpoints

```text
DEV-W
HuggingFaceTB/SmolLM2-135M
revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2
official class LlamaForCausalLM
pinned dtype BF16
pinned eager attention/reference runtime

TARGET-W
meta-llama/Meta-Llama-3.1-405B-Instruct
revision f9801cba95a53242b3cc928a4a418d12571d1c5f
```

DEV-W is an implementation and falsification checkpoint. It is not a 405B performance surrogate.

## 7. Frozen execution workload

Reuse the already committed fixed-public workload:

- six families;
- 128 decode transitions per family;
- greedy and the registered deterministic cases;
- exact reference token comparison;
- exact successor cache/RNG-state comparison;
- official Hugging Face load and reference generation.

No prompt, tile size, lowering mode, or threshold is selected after the result.

Registered tiles:

```text
joint gate/up fiber rows = 128
down output rows         = 128
Q4 fiber screen block    = 32 coefficients
```

## 8. Resource equations

For hidden width `H`, intermediate width `I`, fiber tile `R_f`, and output tile `R_o`, the initial lowering charges:

```text
cold artifact bytes
    = lossless bytes of joint gate/up tiles
    + lossless bytes of down output tiles
    + manifest, hashes, maps, and ABI evidence

peak decoded weight bytes
    = max(2 * R_f * H, R_o * I) * dtype_bytes

native arithmetic
    = gate affine work
    + up affine work
    + SiLU and product
    + down affine work
```

A stacked eligible gate/up call changes call count and memory traversal, not the algebraic MAC count. It is an auxiliary physical rewrite, not the target-scale global solution.

The final target-equivalent p50 fraction remains:

```text
1.2 * 4 / 405 = 1.185185185...%
```

A nonlinear compiler rewrite may be promoted as a core only when its complete artifact, operations, traffic, selector, verification, miss, state, and fallback equations all fit this fraction at target scale. The compiler substrate receives no exemption.

## 9. Gates

### Compiler-substrate pass

All are mandatory:

```text
actual named DEV-W loaded through official runtime
complete real Transformer layer replaced
compile-time selected ABI mode passes every frozen bitwise probe
6 * 128 consecutive token transitions match
6 * 128 successor states match
zero future-token access
zero integrity/control failure
artifact and physical resource trace recorded
remote raw evidence and hashes committed
```

### Core-rewrite promotion

In addition to the substrate pass, at least one automatically generated nonlinear rewrite must demonstrate:

```text
bitwise native reference equality on the frozen Gate
fully charged optimistic operations <= 1.185185185%
fully charged optimistic traffic    <= 1.185185185%
hot checkpoint-dependent state      <= 8 GiB
no hidden dense discovery or fallback
credible scaling reason toward TARGET-W
```

The initial joint-payload lowering is not counted as satisfying this Gate.

## 10. Fixed outcomes

```text
PROMOTE_EXACT_GLOBAL_NONLINEAR_REWRITE
    substrate passes and a target-scale rewrite passes every core threshold

GLOBAL_NONLINEAR_COMPILER_SUBSTRATE_ESTABLISHED_NO_CORE_REWRITE
    exact whole-function compiler and dynamic state pass, but no target-scale rewrite exists in the implemented rule set

REJECT_GLOBAL_COMPILER_NATIVE_ABI
    compile probes or transition/state replay mismatch

INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_DECISION
    official checkpoint or hosted runtime cannot execute
```

A substrate-only pass authorizes extending proof-carrying rewrite rules. It does not authorize a 405B runtime, CUDA kernel, or 4B-class speed claim.

## 11. Strongest current falsification

The strongest checkpoint-specific counterexample is a population of unique, non-antipodal nonlinear fibers whose favorable Q4 tree lower bound already exceeds the complete target fraction. In that case, duplicate/CSE/tree rewrites cannot be the core even though the compiler itself remains useful.

The strongest numerical counterexample is any registered transition where a stacked or tiled lowering changes one BF16 value, token, KV entry, or RNG successor. That lowering fails closed.

## 12. Claim boundary

A hosted pass establishes only a real proof-carrying whole-SwiGLU compiler substrate on DEV-W at the highest actually passed gate. It does not establish:

```text
arbitrary-checkpoint nonlinear compressibility
TARGET-W tensor structure
405B execution
8 GiB peak VRAM
CUDA or PCIe behavior
same-machine native 4B p50/p95 performance
E6 or E7
```

All negative structural results are retained as rewrite-family closures, not as proof that the global compiler research program is impossible.
