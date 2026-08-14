# EXP-075 — Native MTP Checkpoint and Runtime Surface Audit

## Question

Does the smallest official unchanged Qwen3.5 checkpoint contain a complete
native MTP tensor surface, and does one pinned public runtime expose a matching
causal speculative-decoding loader, before any model weights are downloaded?

This is Phase A/B static public-metadata evidence with an E1 ceiling. It is an
auxiliary prerequisite for the revised EXP-074 surrogate, not a new universal
dense-405B mechanism.

## E0 efficiency triage

The target operation is a future target forward over multiple candidate
positions. A checkpoint-native MTP head is the new causal proposal information
source; unchanged-target verification and exact correction remain mandatory.
If accepted blocks are long enough, one target weight stream can be amortized
over many tokens, giving an in-principle order-of-magnitude traffic change.

The current evidence does not support the required acceptance. EXP-074 requires
at least 9 perfect tokens under an unrealistically free fixed-route ceiling,
25 with 0.8B-equivalent proposal work, and 98 under independent-uniform route
union. Dense 405B still requires 85 tokens with a free proposer or 507 with a
4B proposer. Therefore metadata presence is only the cheapest kill Gate.

This continuation changes the evidence basis from assumed MTP availability to
exact public checkpoint keys and exact runtime source mappings. It does not
reopen rejected target-only fixed-point or target-independent drafting.

## Registered sources

Checkpoint:

```text
Qwen/Qwen3.5-0.8B
revision 2fc06364715b967f1860aea9cf38778875588b17
```

Runtime:

```text
vllm-project/vllm
revision a07086e4032e66aacae60ac2fc01e738096e9569
```

Only six bounded UTF-8 files may be fetched: checkpoint `config.json`,
`model.safetensors.index.json`, `README.md`, and three pinned vLLM Python source
files. URLs ending in a weight payload extension fail closed.

## Surface contract

The checkpoint Gate requires:

```text
architecture Qwen3_5ForConditionalGeneration
text model type qwen3_5_text
mtp_num_hidden_layers == 1
exact registered set of 15 mtp.* keys in the safetensors index
all MTP keys mapped to a declared shard
official model-card MTP training and serving markers
```

The runtime Gate requires static evidence that the pinned vLLM source:

```text
maps qwen3_5/qwen3_5_moe to Qwen3_5MTP/Qwen3_5MoeMTP
uses the target checkpoint as the MTP draft source
inherits target quantization configuration
registers both MTP classes
constructs the native predictor from mtp_num_hidden_layers
cycles speculative step indexes over available MTP layers
remaps checkpoint mtp.* names into the runtime MTP module
```

Marker presence is a bounded static interface audit, not proof that the runtime
executes correctly.

## Success and rejection

Pass decision:

```text
PROMOTE_TO_PINNED_QWEN35_08B_ACCEPTED_PREFIX_GATE
```

This authorizes only a separately preregistered small-checkpoint Gate. It does
not itself authorize a 35B/122B download or a page scheduler.

Failure decisions:

```text
REJECT_NATIVE_MTP_SURFACE_UNAVAILABLE
REVISE_RUNTIME_MTP_SURFACE_NOT_EXPOSED
INVALID_METADATA_AUDIT_CONTROL_FAILURE
```

## Twelve-question boundary

1. Amortized operation: unchanged target verification across candidate tokens.
2. New information: checkpoint-native causal MTP state.
3. Selector cost: not measured; must be fully charged next.
4. Wrong proposal detection: exact longest-prefix target verification.
5. Fallback: exact first correction and sequential completion or abort.
6. Output safety: no unverified proposal is committed.
7. Scaling: only block amortization can improve with accepted length; unverified.
8. Sequential dependency: MTP recursively proposes without target future tokens;
   actual causal implementation remains unexecuted.
9. Movement: only metadata moves in EXP-075; all model traffic is unverified.
10. 405B minimum: EXP-074 retains 85/507-token dense requirements.
11. Gap: metadata presence closes none of the 1.185185% performance gap.
12. Falsification: missing keys/loader rejects immediately; later, insufficient
    accepted-prefix p50/p95 rejects the retained branch.

## Strongest counterexamples and stop rule

- configuration advertises MTP but the checkpoint omits one or more MTP keys;
- runtime recognizes the target but silently ignores MTP weights;
- repeated use of one MTP layer yields short or zero accepted prefixes;
- hybrid Gated DeltaNet rollback makes rejection state incorrect;
- proposal, LM-head, verification, rollback, or fallback erases amortization;
- Qwen-specific behavior does not transfer to arbitrary dense checkpoints.

Before this Gate is frozen, prohibit model-weight download, inference, runtime
installation, server commands, CUDA work, expert traces, and page scheduling.

Required interpretation:

> Structurally valid conditions were established. Large-model performance
> remains unverified.

## Authoritative result

The pinned network audit fetched six bounded UTF-8 metadata/source files totaling
`255,779` bytes. No URL addressed a safetensors payload.

Measured static surface:

```text
checkpoint MTP tensors indexed                 15 / 15
checkpoint mtp_num_hidden_layers                     1
vLLM source files audited                       3 / 3
registered controls                             5 / 5
control failures                                    0
declared complete checkpoint bytes      1,746,882,752
```

Both checkpoint and runtime source Gates passed. The declared checkpoint size
is `1.6269113421 GiB`; this is index metadata, not a downloaded byte count.

Decision:

```text
PROMOTE_TO_PINNED_QWEN35_08B_ACCEPTED_PREFIX_GATE
```

Authority: `results/exp_075/summary.json`; preregistration/source commit
`f85ac583a129070247992987d1b3c63634e6447f`; evidence commit
`2fcf7315cf9da491a5ca361536eb0f07e325e74c`; deterministic core SHA-256
`2515bb53a0e2967cfd23e15d18720142337c7d25dc658db057e86e1aa45b5674`.

Interpretation: the earlier MTP-availability assumption is now supported at the
static interface level. Proposal acceptance, recursive-step correctness,
hybrid-state rollback, quantization preservation, actual runtime compatibility,
and every performance quantity remain unverified. No model weight or server
command was used.
