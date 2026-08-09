# EXP-083B Post-hoc Necessary-Condition Audit

- Date: 2026-08-09 Asia/Seoul
- Status: DERIVED AFTER THE FROZEN DECISION; NON-AUTHORITATIVE DIAGNOSTIC
- Authoritative decision: `REJECT_CAUSAL_RESIDUAL_ATLAS_LEGAL_PAIR_OUTWARD_PATH`

## Purpose

The frozen EXP-083B certificate returned an extremely large final RMSNorm
radius after the pre-normalization ball crossed the origin. This audit asks a
narrow question: could merely tightening that implementation-error term rescue
the already observed row under the same global L2-ball/LM-row-norm certificate?

It does not alter the Gate, rerun a prompt, choose another page, or authorize a
new experiment. Values are derived from
`results/exp_083b/raw/prompt_arrays/legal_holdout_english_01.npz` and the
static LM-head row bounds in the checksummed evidence bundle.

## Observed necessary conditions

```text
candidate pre-RMSNorm L2                         10.709112060498093
selected residual-page L2                         8.719004457488106
unread residual L2                               12.308145586545258
verified beta_W                                   1.326752041578861
unread radius beta_W*||u_unread||                16.329857285002372
complete frozen down radius                      23.420520066618046
actual candidate/native down difference           3.5125591928982423
actual candidate/native final-hidden difference  38.90790804031119
candidate top-1 gap                               5.25
best top-two hidden radius ignoring all rounding  4.197825281093514
actual hidden difference / that limit             9.268586812211456x
candidate/native KL                               0.05158216424853682
```

The unread term alone exceeds the candidate pre-RMSNorm norm, so the frozen
sound input ball necessarily reaches the RMSNorm near-zero region before pair,
decomposition, arithmetic, and native terms are added.

There is also a stronger observation independent of the large implementation
radius. For candidate winner `k` and its highest-logit competitor `j`, the
row-norm certificate requires, even with zero arithmetic rounding,

```text
r_hidden < (logit_k - logit_j) / (||w_k|| + ||w_j||)
         = 4.197825281093514.
```

Any sound ball centered at the candidate and containing the observed native
hidden state needs radius at least `38.90790804031119`. Therefore no tightening
of only the RMSNorm implementation envelope can certify this row under the
same global hidden-ball and per-row LM-head certificate. The gap is at least
`9.2686x` before native logit-rounding terms.

## Scope

This closes the frozen common-spectral, one-page, global-L2-ball Atlas primary
path. It is not a universal lower bound against every possible correlated or
direction-aware certificate. Reopening requires a materially new causal
information source and a new fully charged E0 equation; changing rank, page,
layer, selector score, prompt subset, spectral slack, or numerical tolerance is
not sufficient.
