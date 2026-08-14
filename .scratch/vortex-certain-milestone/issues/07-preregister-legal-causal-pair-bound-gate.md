# Preregister the Legal Causal-Pair and Outward-Bound Gate

Type: task
Status: resolved
Blocked by: Run the Causal Residual Atlas First-Decode Gate

## Question

What is the smallest leakage-safe Gate that can distinguish favorable oracle
page existence from an executable Causal Residual Atlas by requiring all of:

- `Q` and native-numerical `Z=WQ` constructed only from committed prefix
  input/image pairs;
- a page choice that never reads native current outputs or candidate logits;
- an outward-rounded unread-residual certificate for the declared output; and
- fully charged state, construction, selection, exact completion/fallback, and
  failure behavior?

The already observed 18 evaluation prompts may not be used to tune selector or
bound choices. The ticket must freeze inputs, numerical semantics, controls,
thresholds, stop rules, evidence ceiling, and the cheapest failure before
assigning another experiment number or authorizing E2/hardware work.

## Answer

Freeze the
[Legal Pair and Outward-Bound Gate](../../../docs/research/CAUSAL_RESIDUAL_ATLAS_LEGAL_PAIR_OUTWARD_GATE.md)
on a new SHA-pinned 24-prompt population. The prior 18 prompts and all their
candidate logits are quarantined. The cheapest executable slice is the first
post-prefill layer-23 `down_proj`, after which only residual addition, final
RMSNorm, and the tied LM head remain.

Two-pass causal Gram-Schmidt derives stored `Q_hat/Z_hat` and a pair-image
defect from committed prefix pairs only. Selection removes the 64-column page
with maximum current residual energy and has no weight/output/logit argument.
A verified `beta_W >= ||W||_2` encloses every unread column; outward native
arithmetic, RMSNorm, and LM-head row margins certify the strict BF16 greedy
winner or execute immutable dense completion.

The favorable fully charged component equation is `1.093706272%` traffic and
`0.928746620%` operations at an explicit 20M-token static compile lifetime.
It requires `99.908521087%` coverage, hence 24/24 tokens, 4/4 per family, zero
false accept, and zero fallback. Ten reference tests pass. No checkpoint row
or experiment number exists; the next ticket may execute only this Gate.
