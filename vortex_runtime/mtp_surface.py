"""Static audit helpers for checkpoint-native MTP surfaces.

The audit consumes only configuration, weight-index, model-card, and runtime
source text.  It never opens a safetensors payload and cannot establish model
acceptance, correctness, latency, or hardware compatibility.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


class MtpSurfaceAuditError(ValueError):
    """Raised when a metadata surface violates the registered contract."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256_bytes(encoded)


def validate_metadata_sources(sources: Sequence[Mapping[str, Any]]) -> None:
    """Fail closed if a registered source could fetch a weight payload."""
    if not sources:
        raise MtpSurfaceAuditError("at least one metadata source is required")
    names: set[str] = set()
    for source in sources:
        name = str(source.get("name", ""))
        output_name = str(source.get("output_name", ""))
        url = str(source.get("url", ""))
        if not name or not output_name or not url.startswith("https://"):
            raise MtpSurfaceAuditError("source name, output name, and HTTPS URL required")
        if name in names:
            raise MtpSurfaceAuditError(f"duplicate source name: {name}")
        names.add(name)
        lowered = url.lower().split("?", 1)[0]
        if lowered.endswith((".safetensors", ".bin", ".gguf", ".pt", ".pth")):
            raise MtpSurfaceAuditError(f"weight payload URL is prohibited: {name}")
        if "/resolve/" in lowered and not lowered.endswith(
            (".json", ".md", ".py", ".txt", ".yml", ".yaml")
        ):
            raise MtpSurfaceAuditError(f"unbounded resolve URL is prohibited: {name}")


def audit_checkpoint_surface(
    *,
    model_config: Mapping[str, Any],
    weight_index: Mapping[str, Any],
    model_card: str,
    expected_architecture: str,
    expected_model_type: str,
    expected_mtp_layers: int,
    expected_mtp_keys: Sequence[str],
    required_card_markers: Sequence[str],
) -> dict[str, Any]:
    """Audit that a pinned checkpoint declares and indexes its native MTP head."""
    if expected_mtp_layers <= 0:
        raise MtpSurfaceAuditError("expected MTP layers must be positive")
    architectures = model_config.get("architectures")
    text_config = model_config.get("text_config")
    weight_map = weight_index.get("weight_map")
    metadata = weight_index.get("metadata")
    if not isinstance(architectures, list) or not isinstance(text_config, Mapping):
        raise MtpSurfaceAuditError("invalid model configuration shape")
    if not isinstance(weight_map, Mapping) or not isinstance(metadata, Mapping):
        raise MtpSurfaceAuditError("invalid weight-index shape")
    total_size = metadata.get("total_size")
    if not isinstance(total_size, int) or total_size <= 0:
        raise MtpSurfaceAuditError("weight index lacks a positive total_size")

    mtp_keys = sorted(str(key) for key in weight_map if str(key).startswith("mtp."))
    expected = sorted(str(key) for key in expected_mtp_keys)
    missing = sorted(set(expected) - set(mtp_keys))
    unexpected = sorted(set(mtp_keys) - set(expected))
    shard_names = sorted({str(weight_map[key]) for key in mtp_keys})
    card_missing = [marker for marker in required_card_markers if marker not in model_card]
    checks = {
        "architecture_matches": architectures == [expected_architecture],
        "model_type_matches": text_config.get("model_type") == expected_model_type,
        "mtp_layer_count_matches": text_config.get("mtp_num_hidden_layers")
        == expected_mtp_layers,
        "mtp_keys_exact": not missing and not unexpected and bool(mtp_keys),
        "mtp_keys_have_shards": bool(shard_names)
        and all(bool(weight_map[key]) for key in mtp_keys),
        "model_card_markers_present": not card_missing,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "architecture": architectures,
        "model_type": text_config.get("model_type"),
        "mtp_num_hidden_layers": text_config.get("mtp_num_hidden_layers"),
        "mtp_use_dedicated_embeddings": text_config.get(
            "mtp_use_dedicated_embeddings"
        ),
        "mtp_tensor_count": len(mtp_keys),
        "mtp_tensor_keys": mtp_keys,
        "missing_mtp_tensor_keys": missing,
        "unexpected_mtp_tensor_keys": unexpected,
        "mtp_shards": shard_names,
        "declared_weight_bytes": total_size,
        "declared_weight_gib": total_size / 1024**3,
        "missing_model_card_markers": card_missing,
    }


def audit_runtime_surface(
    *,
    source_text: Mapping[str, str],
    required_markers: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Audit registered runtime source markers at a pinned source revision."""
    file_rows: list[dict[str, Any]] = []
    for source_name, markers in required_markers.items():
        text = source_text.get(source_name)
        if text is None:
            raise MtpSurfaceAuditError(f"missing runtime source: {source_name}")
        missing = [marker for marker in markers if marker not in text]
        file_rows.append(
            {
                "source": source_name,
                "required_marker_count": len(markers),
                "missing_markers": missing,
                "passed": not missing,
            }
        )
    return {"files": file_rows, "passed": all(row["passed"] for row in file_rows)}


def surface_decision(
    *, checkpoint_passed: bool, runtime_passed: bool, controls_passed: bool
) -> str:
    if not controls_passed:
        return "INVALID_METADATA_AUDIT_CONTROL_FAILURE"
    if not checkpoint_passed:
        return "REJECT_NATIVE_MTP_SURFACE_UNAVAILABLE"
    if not runtime_passed:
        return "REVISE_RUNTIME_MTP_SURFACE_NOT_EXPOSED"
    return "PROMOTE_TO_PINNED_QWEN35_08B_ACCEPTED_PREFIX_GATE"


def require_finite_metrics(payload: Any) -> None:
    """Reject NaN/Inf anywhere in a JSON-like evidence payload."""
    if isinstance(payload, float) and not math.isfinite(payload):
        raise MtpSurfaceAuditError("non-finite evidence metric")
    if isinstance(payload, Mapping):
        for value in payload.values():
            require_finite_metrics(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            require_finite_metrics(value)
