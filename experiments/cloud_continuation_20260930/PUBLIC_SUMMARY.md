# Continuation results: bounded findings, not mission completion

## Executed checks

The unchanged 14 native-transition controls and 16 EXP100A focused tests passed.
The original canonical 3,510-file manifest was unchanged. A pinned public
SmolLM2-135M checkpoint was acquired and verified against its existing SHA256.
The EXP100A runner classification repair is in commit
8aa1f22e0dfa6fc5aa2a3853101de2614ef2fef5; it does not alter historical results.

## Full CPU replay

An unchanged small-model replay subsequently ran in Linux. Graph, cache-codec and
demand candidates matched their same-runtime reference on the registered corpus.
The original frozen Windows comparator nevertheless failed: 3,064/3,510 files
were different, including genuine logits and KV bit differences. Across reference
steps, 470,979/589,824 logit coordinates and 503,174/760,320 KV coordinates differed.
The three sampled token sequences still matched. The exact platform/kernel cause
is not isolated; these observations are not proof of universal portability.
Original baseline files and comparator expectations were not changed.

The attempted supplemental preregistration append failed due to a working-directory
path error and the shell continued into the replay. That deviation was recorded
after start, not claimed as prospectively saved. The underlying original replay
protocol, inputs and comparator were unchanged; this was not a new core selection.
No target speed or memory claim follows from this reproduction check.

## Research frontier

[The construction screen](CONSTRUCTION_SCREEN.md) separates three finite but
unqualified realizations: native accumulator transition composition, globally
adaptive word encoding, and continuation-equivalence compilation. Each still
lacks a cheap, non-enumerative heterogeneous effect producer with complete costs.
No qualifying >=10x route was found in this round. This does not prove that every
possible execution algorithm fails. No expensive experiment is justified merely
by renaming the missing primitive.

## Evidence and completion limits

Raw replay artifacts remain in the active research workspace. Publishing their
binary archive to the public repository was blocked; it is NOT remotely preserved
by this text-only summary. The prior code/control commit is remotely verified.
This distinction must survive handoff. The text contains aggregate experimental
findings and original theoretical calculations, not the blocked raw payload.

THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
THREE_QUALIFYING_NEW_PRINCIPLES=false
FULL_MISSION_O1_O6=OPEN
RAW_REPLAY_HANDOFF=BLOCKED

Remaining work is the constructive native producer/cost proof and permitted
raw-evidence persistence. No 405B/8-GiB/native-4B-Q4/TTFT achievement is claimed.
