"""VORTEX runtime research prototype.

Public symbols are loaded lazily so lightweight, standard-library research
primitives do not require optional Torch/safetensors dependencies merely to
import the package. Existing public names remain available through
``from vortex_runtime import <name>``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AtlasStats": (".atlas_linear", "AtlasStats"),
    "OnlineAtlasLinear": (".atlas_linear", "OnlineAtlasLinear"),
    "ProgressiveLinear": (".progressive", "ProgressiveLinear"),
    "CertificationResult": (".progressive", "CertificationResult"),
    "HuggingFaceLayout": (".hf_loader", "HuggingFaceLayout"),
    "TensorLocator": (".hf_loader", "TensorLocator"),
    "LlamaConfig": (".llama", "LlamaConfig"),
    "StreamingLlama": (".llama", "StreamingLlama"),
    "CPTCConfig": (".cptc", "CPTCConfig"),
    "CPTCError": (".cptc", "CPTCError"),
    "CPTCResult": (".cptc", "CPTCResult"),
    "certify_sum_sign": (".cptc", "certify_sum_sign"),
    "exact_reference": (".cptc", "exact_reference"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
