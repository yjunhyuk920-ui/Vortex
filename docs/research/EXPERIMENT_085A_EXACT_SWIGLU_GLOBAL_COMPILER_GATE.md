# EXP-085A — Exact Whole-SwiGLU Global Nonlinear Compiler Gate

Status before execution: **PREREGISTERED / NO RESULT**  
Evidence ceiling: E1 actual-public-checkpoint structural and native-ABI Gate

## Question

Can an automatic compiler stop treating `gate_proj`, `up_proj`, and
`down_proj` as independent matrix operations and instead compile the complete
SwiGLU function

\[
F_W(x)=W_d\left[\operatorname{SiLU}(W_gx)\odot(W_ux)\right]
\]

into a materially smaller exact executable program for a real unchanged public
checkpoint?

This Gate is intentionally earlier than a full dynamic executor. The theory is
falsified on an actual checkpoint before more streaming, state, kernel, or
target-hardware work.

## Mechanism fingerprint

```text
global_swiglu_atom_dag_exact_equivalence_cse_v1
```

For intermediate neuron `i`, define the complete nonlinear atom

\[
a_i(x)=\operatorname{SiLU}(g_i^T x+b^g_i)(u_i^T x+b^u_i)
\]

and the whole function

\[
F_W(x)=b_d+\sum_i d_i a_i(x).
\]

The compiler hashes exact dtype/shape/byte representations and constructs one
functional DAG containing:

- unique gate affine forms;
- unique up affine forms;
- unique `(gate, up)` nonlinear atoms;
- original-order down accumulation;
- exact duplicate, zero-column, and signed/antipodal diagnostics.

The strict executable may reuse only byte-identical affine forms. It may not
change native accumulation order merely because two real-arithmetic expressions
are algebraically equal.

## Decisive favorable lower bound

Let hidden width be `H`, intermediate width `I`, and output width `O`. The
reference MAC count is

```text
C_ref = 2 I H + O I.
```

If `G`, `U`, and `A` are the numbers of unique exact gate rows, up rows, and
complete nonlinear atoms, the strict count is

```text
C_strict = G H + U H + O I.
```

The decisive optimistic algebraic count additionally grants every duplicate
nonlinear atom a free merge of all corresponding down columns:

```text
C_oracle = G H + U H + O A.
```

This grant ignores native floating reduction-order compatibility, metadata,
loads, stores, scalar nonlinear work, addressing, verification, and code size.
It can only help the candidate. Therefore

```text
C_oracle / C_ref > 1.2 * 4 / 405
```

is a decisive rejection of exact equality/sign/CSE as the missing core.

## Native ABI diagnostic

The executable compiler also emits a whole-function program. Its
`joint_gate_up` lowering concatenates the rows of `Wg` and `Wu` and executes one
linear call, followed by the original SiLU/product/down order. This candidate
is accepted only if every frozen actual-prompt and adversarial BF16 probe is
bitwise equal to the official MLP. An ABI pass cannot rescue a failed
structural resource Gate.

## Frozen public checkpoint and population

```text
DEV-W     HuggingFaceTB/SmolLM2-135M
revision  93efa2f097d58c2a74874c7e644dbc9b0cee75a2
loader    transformers.LlamaForCausalLM.from_pretrained
runtime   torch 2.5.1+cpu / transformers 4.46.3 / safetensors 0.4.5
scope     every SwiGLU MLP layer in the unchanged checkpoint
```

The six already frozen fixed-public workload families supply actual layer-0
MLP inputs. Deterministic zero, sign, scale, random, and coordinate probes are
added before results.

## Controls

- a structured positive control with exact duplicate nonlinear atoms,
  antipodal gate rows, and a zero down column must be recognized;
- a seeded dense-random negative control must not acquire false equivalences;
- a reference-order whole-function program must be bitwise exact;
- nonfinite weights must fail closed;
- repeated compilation must produce an identical plan hash.

Any control failure invalidates the run rather than rejecting the mechanism.

## Promotion Gate

All must hold:

```text
actual official checkpoint loaded
all registered layers audited
all controls pass
optimistic algebraic operation fraction <= 1.185185185%
zero joint-gate/up native-ABI mismatch
finite deterministic plan and result hashes
```

A pass authorizes only a complete-MLP operation-replacement Gate. It does not
authorize a Transformer-layer executor, 405B scaling claim, CUDA kernel, or
8-GiB result.

## Rejection and stop rule

```text
REJECT_EXACT_SWIGLU_EQUIVALENCE_CSE_AS_CORE
```

fires when the favorable algebraic count exceeds the final fraction. On this
result, do not build a full executor or sweep hash definitions, layers, prompts,
tolerances, tile sizes, or row signs around the same equivalence source.

A continuation must introduce a materially different nonlinear representation
that shares generic, non-identical atoms or obtains paid query-time information.
Renaming gate/up fusion, adding a backend, or changing page size does not reopen
this class.

## Claim boundary

This is a proof-first actual-checkpoint compiler Gate. It does not execute
TARGET-W, establish peak GPU VRAM, measure target SSD/PCIe/CUDA, or meet E6/E7.
