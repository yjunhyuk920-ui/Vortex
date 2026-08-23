# EXP-105A — Activation-Ordered Exact Residual-Bound Head Tournament Gate

## Hidden premise reversed

The complete vocabulary head need not be scored eagerly. For the current causal draft activation, read high-magnitude activation dimensions first, update exact integer partial logits, and use exact residual norms to eliminate rows that can no longer win.

## Registered finite-word ABI

- weights: signed Q4 integers `[-8,7]`, group-128 scale/zero bytes charged;
- activations: signed Q8 integers;
- accumulation: exact int64;
- query order: descending `|x_j|`, stable dimension-ID tie break;
- winner tie: lowest token ID;
- bound: `|w_tail dot x_tail| <= ceil(sqrt(||w_tail||^2 ||x_tail||^2))`;
- prune only when `upper < best_lower`.

The runtime returns a winner without evaluating the dense reference. A separate validation function performs the complete dot product only to verify the runtime winner; reference traffic is not counted as mechanism traffic and cannot influence selection.

## Fully charged query ledger

Charge:

- every selected Q4 payload nibble;
- every newly touched group-128 scale/zero;
- initial row norms;
- Q8 activation and dimension order;
- int64 partial-score read/write;
- uint64 residual-norm read/write;
- active row IDs/flags;
- one complete realistic-Q4 target-width draft layer;
- resident capacity and fixed EXP-104A reserves.

No future target state, static top-k, free selector state, free HBM scan, free metadata, or unmeasured compression is allowed.

## Three principles screened

1. activation-ordered exact residual-bound head index — implemented;
2. one-scan multi-token resident transducer — structural causal dependency Gate;
3. cross-matrix decision-bit program — not reopened without a new information source beyond the closed self-contained program family.

## Promotion Gate

A complete causal draft path must fit 8 GiB and remain at or below `2,400,000,000` bytes per proposed/committed token before target verification. A head-only query without a concrete causal state source or measured accepted continuation is incomplete and cannot pass.

## Stop rule

On rejection, do not retune block size, dimension order, Cauchy threshold, state-byte accounting, or synthetic distribution. Reopening requires a stronger exact query-time bound/data structure with fully charged metadata and a concrete causal state source cheaper than one complete layer.
