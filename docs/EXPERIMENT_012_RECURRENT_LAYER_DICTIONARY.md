# Experiment 012 — full-depth recurrent layer dictionary

Evidence level: **E2 causal resident-dictionary draft + optimistic target lower bound**

## New execution principle

The 8 GiB constraint prevents a full 405B Q4 representation from residing on
the GPU. A three-layer thinned draft fits, but discards nearly all target depth.
This experiment keeps the full number of forward positions while storing only
three unique target layers:

```text
3 resident target layers
        ↓ reused according to a deterministic schedule
126 executed layer positions
        ↓
full-depth Q4 recurrent draft
```

The original checkpoint remains unchanged. There is no training, adapter,
distillation or learned layer fusion.

## Why this may close the hardware envelope

The main 405B obstacle is weight movement rather than arithmetic. A recurrent
layer dictionary changes the balance:

- embeddings, LM head and three Q4 decoder layers remain resident below 8 GiB;
- all 126 depth positions are still computed;
- the same three weight sets are read repeatedly from local VRAM;
- projected dense arithmetic at 160 TOPS is within the 1.2x native-4B compute
  envelope, before draft-tree and exact-verification overhead.

Thus this is the first candidate in the project that simultaneously projects:

```text
resident weights <= 8 GiB
full target depth retained
no host stream for the draft
raw dense compute near the 4B latency envelope
```

Accuracy is the decisive unknown.

## Schedules

Representative layers are selected directly from the target:

- `uniform:nearest` — first, middle and final representatives assigned to the
  nearest original depth positions;
- `uniform:cyclic` — the same representatives repeated cyclically;
- `edge:nearest` — first two and final layer assigned by nearest depth;
- `front:cyclic` — first three layers repeated cyclically.

TinyLlama keeps the original 22 execution positions but only three unique layer
objects. Q4 quantization is applied once per unique storage.

## Causal tree gate

Each recurrent draft constructs a depth-12, top-32, beam-64 tree under 1024
unique nodes. The exact target continuation is used only to measure whether the
correct path remains alive; it never affects scheduling, generation or pruning.

## Promotion rule

A point survives only when:

```text
405B recurrent draft memory <= 8 GiB
405B recurrent draft compute <= 1.2x native 4B proxy
exact target path survives all tested causal depths
Q4 target-side serialized verification lower bound passes
```

Passing remains necessary but insufficient because Q6/Q8 exact verification,
KV memory, kernel efficiency and measured 8 GiB wall-clock are not yet charged.

## Architectural consequence

If the recurrent draft preserves substantially more causal target path than the
three-layer thinned draft, the next step is a progressive per-position residual
stream:

```text
original layer = resident representative + compressed layer residual
```

Only residual bitplanes required to certify uncertain decisions would be read.
This would combine resident full-depth execution with the existing progressive
precision router rather than returning to full-model weight streaming.
