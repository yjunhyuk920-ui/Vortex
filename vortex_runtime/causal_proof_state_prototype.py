"""PROTOTYPE -- drive the EXP-079A proof-state transitions by hand.

Question: can a deferred exact dependency remain fail-closed while hot bounds,
cold refinements, certification, budget exhaustion, and exact fallback compete?
The terminal shell is disposable; the pure transition functions live in
``vortex_runtime.causal_proof_state``.
"""
from __future__ import annotations

import json

from vortex_runtime.causal_proof_state import (
    CausalProofStateError,
    certify_proof,
    commit_proof,
    exact_fallback,
    install_hot_bound,
    refine_proof,
    start_proof_machine,
)


def main() -> None:
    block_bytes = 160_000
    state = start_proof_machine(
        allowance_bytes=2_400_000,
        total_blocks=12,
        winner_margin=1.0,
    )
    while True:
        print("\033[2J\033[H", end="")
        print("\033[1mCPSM deferred-proof prototype\033[0m")
        print(
            "Question: does the state machine ever commit an uncertified or "
            "over-budget decision?\n"
        )
        print(json.dumps(state.as_dict(), indent=2, sort_keys=True))
        print(
            "\n\033[1m[h]\033[0m hot bound  "
            "\033[1m[r]\033[0m refine  "
            "\033[1m[c]\033[0m certify/commit  "
            "\033[1m[f]\033[0m exact fallback  "
            "\033[1m[q]\033[0m quit"
        )
        action = input("> ").strip().lower()[:1]
        if action == "q":
            return
        try:
            if action == "h":
                state = install_hot_bound(
                    state, sidecar_bytes=480_000, uncertainty=4.0
                )
            elif action == "r":
                state = refine_proof(
                    state,
                    cold_bytes=block_bytes,
                    next_uncertainty=max(0.0, state.remaining_uncertainty * 0.55),
                )
            elif action == "c":
                state = commit_proof(certify_proof(state))
            elif action == "f":
                state = exact_fallback(state)
        except CausalProofStateError as error:
            input(f"\n\033[1mRejected transition:\033[0m {error}\nPress Enter...")


if __name__ == "__main__":
    main()
