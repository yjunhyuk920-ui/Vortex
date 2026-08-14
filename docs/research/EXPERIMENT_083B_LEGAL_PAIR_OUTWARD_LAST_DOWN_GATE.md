# EXP-083B — Legal Pair and Outward-Bound Last-Down Gate

- Date opened: 2026-08-09 Asia/Seoul
- Status: EXECUTED ONCE, INDEPENDENTLY VERIFIED, REJECTED
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

## Authoritative result

The canonical run used source HEAD
`336d59b580104af327a466ad57d7b5c2af9e7a37`, the frozen config SHA-256
`2c8bdf12535e18327f0a4116e8f9dc4b918f900208d405972cfa421764bdd5c1`,
and the unchanged pinned payload. It stopped on the first evaluation row as
required:

```text
prompt                         legal_holdout_english_01
pair rank                      16
selected page                  0
certificate resolved           false
fallbacks                      1
false accepts                  0
control/leakage failures       0 / 0
candidate/native winner        21461 / 21461
candidate-to-native KL         0.05158216424853682
projection radius / actual     23.4205200666 / 3.5125591929
decision                       REJECT_CAUSAL_RESIDUAL_ATLAS_LEGAL_PAIR_OUTWARD_PATH
```

The model-forward-free independent verifier passed all ten evidence checks and
rebuilt deterministic-core SHA-256
`57e78fd4d7b1bdc6e97c705a5acb2e7e408a1023e38ae933799b00b8e3711ff4`.
The checksummed authority is `results/exp_083b`.

The largest frozen local terms were the unread residual radius
`16.3298572850` and pair-image radius `6.2376633562`. A post-hoc
necessary-condition audit shows that even an oracle ball containing the
observed native hidden state needs radius `38.9079080403`, while the top-two
LM-row certificate permits less than `4.1978252811` before rounding. Tightening
only the RMSNorm implementation envelope therefore cannot rescue this row.

## Frozen implementation boundary

The executable implementation is pinned at commit
`ecf753d7352bcca47767d0fe91c46d84cca66b59`; the final config SHA-256 is
`2c8bdf12535e18327f0a4116e8f9dc4b918f900208d405972cfa421764bdd5c1`.
The canonical runner records the exact enclosing source-freeze `HEAD` in its
environment artifact. No prompt forward occurred before this pin.

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

## Pre-execution claim boundary (historical)

At source-freeze commit `336d59b580104af327a466ad57d7b5c2af9e7a37`, no
prompt-conditioned checkpoint result, result directory, expected decision,
hardware action, model download, private-server command, or E2-E7 evidence
existed. Even a pass would have remained a one-position, one-projection E1
certificate falsification with all earlier work dense.

The registered layer-23 matrix-only compiler was preflighted before source
freeze without tokenization or a model forward. The verified-Cholesky proof
accepted in one attempt with `beta_W=1.326752041578861` and strict
positive-definite margin lower bound `1.94850297451582e-07`. This static
checkpoint metadata did not inspect any of the 24 prompts, activations,
candidate outputs, or native outputs and did not change a frozen Gate choice.
