from __future__ import annotations

from dataclasses import asdict, dataclass
from math import floor

import torch

GIB = 1024**3
MIB = 1024**2


@dataclass(frozen=True)
class Top1FamilyShape:
    rows: int
    columns: int
    row_pairs: int
    selector_columns: int
    payload_columns: int
    decision_bits: int
    matrix_parameters: int
    decision_bits_per_parameter: float
    metadata_mib: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class Top1FamilyEnumeration:
    shape: Top1FamilyShape
    enumerated_checkpoints: int
    expected_functions: int
    observed_functions: int
    decoded_tables_match: bool
    all_winners_unique: bool
    minimum_winner_margin: float
    injective: bool

    @property
    def passes(self) -> bool:
        return bool(
            self.enumerated_checkpoints == self.expected_functions
            and self.observed_functions == self.expected_functions
            and self.decoded_tables_match
            and self.all_winners_unique
            and self.minimum_winner_margin > 0
            and self.injective
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["shape"] = self.shape.to_dict()
        payload["passes"] = self.passes
        return payload


@dataclass(frozen=True)
class OperatorShapeBound:
    name: str
    rows: int
    columns: int
    copies: int
    bits_per_copy: int
    total_bits: int
    total_mib: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True)
class OperatorCollectionBound:
    hidden_size: int
    intermediate_size: int
    kv_projection_size: int
    layers: int
    vocabulary_size: int
    include_lm_head: bool
    matrix_bounds: tuple[OperatorShapeBound, ...]
    decoder_layer_bits: int
    decoder_layer_mib: float
    decoder_stack_bits: int
    decoder_stack_gib: float
    lm_head_bits: int
    lm_head_mib: float
    total_bits: int
    total_gib: float
    resident_limit_gib: float
    exceeds_resident_limit: bool
    direct_classifier_bound_proven: bool
    independent_operator_collection_bound_proven: bool
    full_transformer_top1_bound_proven: bool

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["matrix_bounds"] = [item.to_dict() for item in self.matrix_bounds]
        return payload


def top1_family_shape(*, rows: int, columns: int) -> Top1FamilyShape:
    if rows < 2 or columns < 2:
        raise ValueError("family requires at least two rows and two columns")
    pairs = min(floor(rows / 2), floor(columns / 2))
    payload = columns - pairs
    bits = pairs * payload
    parameters = rows * columns
    return Top1FamilyShape(
        rows=rows,
        columns=columns,
        row_pairs=pairs,
        selector_columns=pairs,
        payload_columns=payload,
        decision_bits=bits,
        matrix_parameters=parameters,
        decision_bits_per_parameter=bits / parameters,
        metadata_mib=bits / 8 / MIB,
    )


def bit_table_from_index(
    index: int,
    *,
    row_pairs: int,
    payload_columns: int,
) -> torch.Tensor:
    bits = row_pairs * payload_columns
    if index < 0 or index >= 1 << bits:
        raise ValueError("bit table index is out of range")
    values = [(index >> offset) & 1 for offset in range(bits)]
    return torch.tensor(values, dtype=torch.long).reshape(
        row_pairs,
        payload_columns,
    )


def encode_selector_payload_classifier(
    bit_table: torch.Tensor,
    *,
    rows: int,
    columns: int,
    selector_margin: float = 4.0,
) -> torch.Tensor:
    shape = top1_family_shape(rows=rows, columns=columns)
    bits = bit_table.detach().to("cpu", torch.long)
    if bits.shape != (shape.row_pairs, shape.payload_columns):
        raise ValueError("bit table shape does not match classifier family")
    if not bool(torch.all((bits == 0) | (bits == 1))):
        raise ValueError("bit table must be binary")
    if selector_margin <= 1.0:
        raise ValueError("selector margin must exceed payload advantage")

    weight = torch.zeros(rows, columns, dtype=torch.float64)
    for pair in range(shape.row_pairs):
        row_zero = 2 * pair
        row_one = row_zero + 1
        weight[row_zero, pair] = selector_margin
        weight[row_one, pair] = selector_margin
        for payload in range(shape.payload_columns):
            column = shape.selector_columns + payload
            value = int(bits[pair, payload].item())
            weight[row_zero, column] = 1.0 if value == 0 else 0.0
            weight[row_one, column] = 1.0 if value == 1 else 0.0
    return weight


def selector_payload_queries(*, rows: int, columns: int) -> torch.Tensor:
    shape = top1_family_shape(rows=rows, columns=columns)
    queries = torch.zeros(shape.decision_bits, columns, dtype=torch.float64)
    offset = 0
    for pair in range(shape.row_pairs):
        for payload in range(shape.payload_columns):
            queries[offset, pair] = 1.0
            queries[offset, shape.selector_columns + payload] = 1.0
            offset += 1
    return queries


def top1_signature(
    weight: torch.Tensor,
    *,
    require_unique: bool = True,
) -> tuple[tuple[int, ...], float]:
    source = weight.detach().to("cpu", torch.float64)
    if source.ndim != 2:
        raise ValueError("weight must be a matrix")
    rows, columns = source.shape
    queries = selector_payload_queries(rows=rows, columns=columns)
    logits = queries @ source.T
    values, indices = torch.topk(logits, k=2, dim=1)
    margins = values[:, 0] - values[:, 1]
    minimum_margin = float(margins.min().item())
    if require_unique and minimum_margin <= 0:
        raise RuntimeError("classifier family produced a non-unique winner")
    return tuple(int(value) for value in indices[:, 0].tolist()), minimum_margin


def decode_signature_bits(
    signature: tuple[int, ...],
    *,
    rows: int,
    columns: int,
) -> torch.Tensor:
    shape = top1_family_shape(rows=rows, columns=columns)
    if len(signature) != shape.decision_bits:
        raise ValueError("signature length does not match family")
    decoded = torch.empty(shape.row_pairs, shape.payload_columns, dtype=torch.long)
    offset = 0
    for pair in range(shape.row_pairs):
        row_zero = 2 * pair
        row_one = row_zero + 1
        for payload in range(shape.payload_columns):
            winner = signature[offset]
            if winner == row_zero:
                decoded[pair, payload] = 0
            elif winner == row_one:
                decoded[pair, payload] = 1
            else:
                raise RuntimeError("winner escaped its selected row pair")
            offset += 1
    return decoded


def enumerate_top1_function_family(
    *,
    rows: int,
    columns: int,
    maximum_bits: int = 16,
) -> Top1FamilyEnumeration:
    shape = top1_family_shape(rows=rows, columns=columns)
    if shape.decision_bits > maximum_bits:
        raise ValueError("family is too large for exhaustive enumeration")
    expected = 1 << shape.decision_bits
    signatures: set[tuple[int, ...]] = set()
    decoded_match = True
    all_unique = True
    minimum_margin = float("inf")

    for index in range(expected):
        table = bit_table_from_index(
            index,
            row_pairs=shape.row_pairs,
            payload_columns=shape.payload_columns,
        )
        weight = encode_selector_payload_classifier(
            table,
            rows=rows,
            columns=columns,
        )
        signature, margin = top1_signature(weight)
        decoded = decode_signature_bits(
            signature,
            rows=rows,
            columns=columns,
        )
        decoded_match = decoded_match and bool(torch.equal(decoded, table))
        all_unique = all_unique and margin > 0
        minimum_margin = min(minimum_margin, margin)
        signatures.add(signature)

    return Top1FamilyEnumeration(
        shape=shape,
        enumerated_checkpoints=expected,
        expected_functions=expected,
        observed_functions=len(signatures),
        decoded_tables_match=decoded_match,
        all_winners_unique=all_unique,
        minimum_winner_margin=minimum_margin,
        injective=len(signatures) == expected,
    )


def _operator_bound(
    *,
    name: str,
    rows: int,
    columns: int,
    copies: int,
) -> OperatorShapeBound:
    shape = top1_family_shape(rows=rows, columns=columns)
    total = shape.decision_bits * copies
    return OperatorShapeBound(
        name=name,
        rows=rows,
        columns=columns,
        copies=copies,
        bits_per_copy=shape.decision_bits,
        total_bits=total,
        total_mib=total / 8 / MIB,
    )


def llama_operator_collection_bound(
    *,
    hidden_size: int = 16_384,
    intermediate_size: int = 53_248,
    kv_projection_size: int = 1_024,
    layers: int = 126,
    vocabulary_size: int = 128_256,
    include_lm_head: bool = True,
    resident_limit_gib: float = 8.0,
) -> OperatorCollectionBound:
    if min(
        hidden_size,
        intermediate_size,
        kv_projection_size,
        layers,
        vocabulary_size,
    ) <= 1:
        raise ValueError("operator collection dimensions are invalid")

    per_layer = (
        _operator_bound(name="q_proj", rows=hidden_size, columns=hidden_size, copies=1),
        _operator_bound(name="k_proj", rows=kv_projection_size, columns=hidden_size, copies=1),
        _operator_bound(name="v_proj", rows=kv_projection_size, columns=hidden_size, copies=1),
        _operator_bound(name="o_proj", rows=hidden_size, columns=hidden_size, copies=1),
        _operator_bound(name="gate_proj", rows=intermediate_size, columns=hidden_size, copies=1),
        _operator_bound(name="up_proj", rows=intermediate_size, columns=hidden_size, copies=1),
        _operator_bound(name="down_proj", rows=hidden_size, columns=intermediate_size, copies=1),
    )
    layer_bits = sum(item.total_bits for item in per_layer)
    stack_bits = layer_bits * layers
    matrix_bounds = tuple(
        OperatorShapeBound(
            name=item.name,
            rows=item.rows,
            columns=item.columns,
            copies=layers,
            bits_per_copy=item.bits_per_copy,
            total_bits=item.bits_per_copy * layers,
            total_mib=item.bits_per_copy * layers / 8 / MIB,
        )
        for item in per_layer
    )
    lm_bits = (
        top1_family_shape(rows=vocabulary_size, columns=hidden_size).decision_bits
        if include_lm_head
        else 0
    )
    total = stack_bits + lm_bits
    return OperatorCollectionBound(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        kv_projection_size=kv_projection_size,
        layers=layers,
        vocabulary_size=vocabulary_size,
        include_lm_head=include_lm_head,
        matrix_bounds=matrix_bounds,
        decoder_layer_bits=layer_bits,
        decoder_layer_mib=layer_bits / 8 / MIB,
        decoder_stack_bits=stack_bits,
        decoder_stack_gib=stack_bits / 8 / GIB,
        lm_head_bits=lm_bits,
        lm_head_mib=lm_bits / 8 / MIB,
        total_bits=total,
        total_gib=total / 8 / GIB,
        resident_limit_gib=resident_limit_gib,
        exceeds_resident_limit=total / 8 / GIB > resident_limit_gib,
        direct_classifier_bound_proven=True,
        independent_operator_collection_bound_proven=True,
        full_transformer_top1_bound_proven=False,
    )
