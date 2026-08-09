# EXP-083B — Legal Pair and Outward-Bound Last-Down Gate

- Date opened: 2026-08-09 Asia/Seoul
- Status: PREREGISTERED SOURCE, NOT EXECUTED
- Phase: C small-real-checkpoint certificate falsification
- Evidence ceiling: E1

## Immutable authority

The scientific contract is
`docs/research/CAUSAL_RESIDUAL_ATLAS_LEGAL_PAIR_OUTWARD_GATE.md`, SHA-256
`4cdd19b04b9232e1d2201b93ab1eb05f7f9eba071495713f35a9be588cadb930`,
committed at `f0c17c5646f842cdf2dcb3138e6ba51d92a5c083` before any one of the
24 new prompts was executed against the checkpoint.

EXP-083B may implement that authority but may not change its prompt population
or order, layer, projection, position, rank, page width, pair compiler,
selector, numerical contract, output contract, thresholds, or stop rule after
observing a checkpoint result.

## Frozen implementation boundary

The runner first compiles a static outward Gram certificate for layer-23
`down_proj`. For each prompt it then:

1. runs one unchanged prefill and records committed prefix input/image pairs;
2. commits the unchanged prompt-last greedy token;
3. constructs BF16 `Q_hat/Z_hat` by two-pass causal pair-only MGS;
4. forms the current residual and selects the maximum-energy 64-column page;
5. reads only that page for the candidate `down_proj` result;
6. propagates every pair, decomposition, arithmetic, unread, residual-add,
   final-RMSNorm, and LM-head error term outward;
7. freezes the certificate verdict before evaluator access to native current
   output or logits; and
8. executes immutable dense completion on an unresolved verdict.

The static verifier uses an outward smaller-Gram inclusion, a candidate
Cholesky factor, a verified factor-inverse residual, and a verified factor
residual to prove `beta_W^2 I - G_W > 0`. A numerical eigensolver proposes the
bound but is not trusted by the proof. The registered target equation retains
the larger preregistered verified-compiler charge.

## One-shot stop rule

Any valid row that cannot construct rank 16, cannot prove a finite strict final
winner, or requires fallback immediately returns
`REJECT_CAUSAL_RESIDUAL_ATLAS_LEGAL_PAIR_OUTWARD_PATH`. It may not enumerate a
different page or tune a bound. A certified winner that differs from the
unchanged native winner is a control failure, not a scientific success.

Promotion requires 24/24 tokens, 4/4 in every family, zero false accept, zero
fallback, and mean/p95 target-to-candidate KL at most `0.02/0.05`.

## Claim boundary before execution

No prompt-conditioned checkpoint result, result directory, expected decision,
hardware action, model download, private-server command, or E2-E7 evidence
exists in this source. Even a pass would remain a one-position, one-projection
E1 certificate falsification with all earlier work dense.

The registered layer-23 matrix-only compiler was preflighted before source
freeze without tokenization or a model forward. The verified-Cholesky proof
accepted in one attempt with `beta_W=1.326752041578861` and strict
positive-definite margin lower bound `1.94850297451582e-07`. This static
checkpoint metadata did not inspect any of the 24 prompts, activations,
candidate outputs, or native outputs and did not change a frozen Gate choice.
