"""PROTOTYPE -- interactive EXP-078A construction-amortization calculator."""
from __future__ import annotations

import json

from vortex_runtime.tangent_macroblock import (
    charged_cycle_fraction,
    dense_macroblock_cost,
    minimum_hot_tokens,
)


PROFILES = {
    "Qwen3.5-0.8B dense MLP": (1024, 3584, 1),
    "Qwen3.5-122B-A10B 9-path combined MLP": (3072, 1024, 9),
}


def snapshot(profile_index: int, hot_tokens: int, allowance: float) -> dict[str, object]:
    name = list(PROFILES)[profile_index]
    hidden, intermediate, paths = PROFILES[name]
    costs = dense_macroblock_cost(
        hidden_size=hidden,
        intermediate_size=intermediate,
        active_paths=paths,
    )
    construction = float(costs["materialization_exact_token_equivalents"])
    hot = float(costs["hot_fraction_of_exact"])
    return {
        "profile": name,
        "hot_tokens": hot_tokens,
        "allowance_fraction": allowance,
        "hot_fraction": hot,
        "construction_exact_token_equivalents": construction,
        "charged_cycle_fraction": charged_cycle_fraction(
            hot_tokens=hot_tokens,
            hot_fraction=hot,
            construction_token_equivalents=construction,
        ),
        "minimum_hot_tokens": minimum_hot_tokens(
            allowance_fraction=allowance,
            hot_fraction=hot,
            construction_token_equivalents=construction,
        ),
    }


def main() -> None:
    profile_index = 0
    hot_tokens = 8
    allowance = 0.12
    while True:
        print("\033[2J\033[H", end="")
        print("\033[1mTMR construction-amortization prototype\033[0m")
        print("Question: can an exact frozen MLP macro-operator amortize its direct build cost?\n")
        print(json.dumps(snapshot(profile_index, hot_tokens, allowance), indent=2))
        print("\n\033[1m[m]\033[0m model  \033[1m[+]\033[0m double span  \033[1m[-]\033[0m halve span  \033[1m[a]\033[0m p50/p95 allowance  \033[1m[q]\033[0m quit")
        action = input("> ").strip().lower()[:1]
        if action == "q":
            return
        if action == "m":
            profile_index = (profile_index + 1) % len(PROFILES)
        elif action == "+":
            hot_tokens *= 2
        elif action == "-":
            hot_tokens = max(1, hot_tokens // 2)
        elif action == "a":
            allowance = 0.15 if allowance == 0.12 else 0.12


if __name__ == "__main__":
    main()
