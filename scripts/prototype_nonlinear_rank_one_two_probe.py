"""Exact SMT search for the smallest nonlinear rank-one probe gap.

The source is an arbitrary binary 2x3 matrix (six bits).  The encoder may map
its 64 values injectively to arbitrary seven-bit codewords.  For every
rank-one parity query, the decoder may choose one first bit, choose its second
bit from the first returned value, and use an arbitrary two-bit truth table.

This is strictly broader than a systematic source plus one nonlinear advice
bit.  A SAT witness is a genuine nonlinear two-probe seed.  UNSAT proves that
the first degree-capacity gap is not attainable at these exact parameters.
The script requires the optional ``z3-solver`` research dependency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROWS = 2
COLUMNS = 3
SOURCE_BITS = ROWS * COLUMNS
SOURCE_VALUES = 1 << SOURCE_BITS
STORED_BITS = 7


def rank_one_masks() -> list[int]:
    masks: list[int] = []
    for left in range(1, 1 << ROWS):
        for right in range(1, 1 << COLUMNS):
            mask = 0
            for row in range(ROWS):
                for column in range(COLUMNS):
                    if ((left >> row) & 1) and ((right >> column) & 1):
                        mask |= 1 << (row * COLUMNS + column)
            if mask not in masks:
                masks.append(mask)
    return masks


def parity(value: int) -> bool:
    return bool(value.bit_count() & 1)


def solve(*, timeout_ms: int) -> dict[str, object]:
    try:
        import z3  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "install z3-solver or put it on PYTHONPATH before running"
        ) from error

    masks = rank_one_masks()
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)

    codewords = [
        z3.BitVec(f"code_{source}", STORED_BITS)
        for source in range(SOURCE_VALUES)
    ]
    solver.add(z3.Distinct(*codewords))
    # XORing every codeword by the same constant preserves the model after
    # complementing decoder branches, so this removes one exact symmetry.
    solver.add(codewords[0] == 0)

    first_addresses = [z3.Int(f"first_{query}") for query in range(len(masks))]
    second_addresses = [
        [z3.Int(f"second_{query}_{first}") for first in range(2)]
        for query in range(len(masks))
    ]
    outputs = [
        [
            [
                z3.Bool(f"out_{query}_{first}_{second}")
                for second in range(2)
            ]
            for first in range(2)
        ]
        for query in range(len(masks))
    ]

    def selected_bit(codeword: object, address: object) -> object:
        choices = [
            z3.And(address == bit, z3.Extract(bit, bit, codeword) == 1)
            for bit in range(STORED_BITS)
        ]
        return z3.Or(*choices)

    for query, mask in enumerate(masks):
        solver.add(first_addresses[query] >= 0)
        solver.add(first_addresses[query] < STORED_BITS)
        for first in range(2):
            solver.add(second_addresses[query][first] >= 0)
            solver.add(second_addresses[query][first] < STORED_BITS)

        for source, codeword in enumerate(codewords):
            first_value = selected_bit(codeword, first_addresses[query])
            second_address = z3.If(
                first_value,
                second_addresses[query][1],
                second_addresses[query][0],
            )
            second_value = selected_bit(codeword, second_address)
            decoded = z3.If(
                first_value,
                z3.If(
                    second_value,
                    outputs[query][1][1],
                    outputs[query][1][0],
                ),
                z3.If(
                    second_value,
                    outputs[query][0][1],
                    outputs[query][0][0],
                ),
            )
            solver.add(decoded == parity(source & mask))

    started = time.perf_counter()
    status = solver.check()
    elapsed = time.perf_counter() - started
    payload: dict[str, object] = {
        "classification": "E0_NONLINEAR_2X3_TWO_PROBE_SMT",
        "rows": ROWS,
        "columns": COLUMNS,
        "source_bits": SOURCE_BITS,
        "stored_bits": STORED_BITS,
        "rank_one_query_count": len(masks),
        "adaptive_probe_limit": 2,
        "encoder_is_arbitrary_injective": True,
        "decoder_second_address_depends_on_first_value": True,
        "decoder_output_table_is_arbitrary": True,
        "timeout_ms": timeout_ms,
        "solver_status": str(status),
        "elapsed_seconds": elapsed,
    }

    if status == z3.sat:
        model = solver.model()
        encoded = [model.eval(word).as_long() for word in codewords]
        algorithms: list[dict[str, object]] = []
        mismatches = 0
        for query, mask in enumerate(masks):
            first = model.eval(first_addresses[query]).as_long()
            second = [
                model.eval(second_addresses[query][value]).as_long()
                for value in range(2)
            ]
            table = [
                [
                    bool(z3.is_true(model.eval(outputs[query][a][b])))
                    for b in range(2)
                ]
                for a in range(2)
            ]
            for source, codeword in enumerate(encoded):
                first_value = (codeword >> first) & 1
                second_value = (
                    codeword >> second[first_value]
                ) & 1
                answer = table[first_value][second_value]
                mismatches += answer != parity(source & mask)
            algorithms.append(
                {
                    "mask": mask,
                    "first_address": first,
                    "second_address_by_first_value": second,
                    "output_table": table,
                }
            )
        payload.update(
            {
                "codewords": encoded,
                "algorithms": algorithms,
                "verification_mismatches": mismatches,
                "decision": "FOUND_NONLINEAR_TWO_PROBE_SEED",
            }
        )
    elif status == z3.unsat:
        payload["decision"] = "REJECT_ARBITRARY_7BIT_TWO_PROBE_ENCODING"
    else:
        payload["decision"] = "INCONCLUSIVE_SOLVER_TIMEOUT_OR_UNKNOWN"
        payload["reason_unknown"] = solver.reason_unknown()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-ms", type=int, default=600_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--z3-path", type=Path)
    args = parser.parse_args()
    if args.z3_path is not None:
        sys.path.insert(0, str(args.z3_path.resolve()))
    payload = solve(timeout_ms=args.timeout_ms)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
