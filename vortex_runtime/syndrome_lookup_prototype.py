"""Throwaway TUI for driving the EXP-081A recovery state by hand."""

from __future__ import annotations

import argparse
import json
import random

from .syndrome_lookup import add, compile_random, direct_matvec, execute_query, matvec


def fixture() -> tuple[object, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    p = 2_147_483_647
    rng = random.Random(8101)
    weight = tuple(tuple(rng.randrange(101) for _ in range(5)) for _ in range(7))
    dictionary = tuple(tuple(rng.randrange(101) for _ in range(2)) for _ in range(7))
    compiled = compile_random(
        weight=weight,
        dictionary=dictionary,
        verification_rows=3,
        prime=p,
        seed=8102,
    )
    x = (3, 1, 4, 1, 5)
    exact = direct_matvec(weight, x, p)
    code_error = matvec(compiled.dictionary, (7, 11), p)
    in_code = tuple((a - b) % p for a, b in zip(exact, code_error))
    out_code = list(in_code)
    out_code[0] = (out_code[0] + 1) % p
    return compiled, x, in_code, tuple(out_code)


def render(mode: str, result: object | None) -> None:
    print("\033[2J\033[H", end="")
    print("\033[1mEXP-081A syndrome lookup prototype\033[0m")
    print("\033[2mthrowaway finite-field state; not a production runtime\033[0m\n")
    print(f"\033[1mmode\033[0m: {mode}")
    print("\033[1mresult\033[0m:")
    print(json.dumps(None if result is None else result.to_dict(), indent=2))
    print("\n\033[1m[i]\033[0m in-code  \033[1m[o]\033[0m out-of-code  "
          "\033[1m[f]\033[0m fault  \033[1m[r]\033[0m run  \033[1m[q]\033[0m quit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args()
    compiled, x, in_code, out_code = fixture()
    if args.batch:
        for mode, candidate in (("in-code", in_code), ("out-of-code", out_code)):
            result = execute_query(compiled, x, candidate)
            print(mode, json.dumps(result.to_dict(), sort_keys=True))
        return
    mode, candidate, result = "in-code", in_code, None
    while True:
        render(mode, result)
        action = input("> ").strip().lower()[:1]
        if action == "q":
            return
        if action == "i":
            mode, candidate = "in-code", in_code
        elif action == "o":
            mode, candidate = "out-of-code", out_code
        elif action == "f":
            mode = "fault"
            candidate = add(in_code, (0, 1, 0, 0, 0, 0, 0), compiled.prime)
        elif action == "r":
            result = execute_query(compiled, x, candidate)


if __name__ == "__main__":
    main()
