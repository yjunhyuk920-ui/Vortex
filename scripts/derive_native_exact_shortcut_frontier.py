from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_runtime.native_exact_shortcut_frontier import derive_audit


DEFAULT_MODEL = Path(
    ".deps/exp076-model/model.safetensors-00001-of-00001.safetensors"
)
DEFAULT_EVIDENCE = Path(
    "results/exp_083b/raw/prompt_arrays/legal_holdout_english_01.npz"
)
TENSOR_NAME = "model.language_model.layers.23.mlp.down_proj.weight"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_weight(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(TENSOR_NAME)
    values = tensor.float().numpy()
    words = tensor.view(torch.int16).numpy().view(np.uint16)
    return values, words


def write_result(output_dir: Path, payload: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = output_dir / "summary.json"
    summary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "checksums.sha256").write_text(
        f"{_sha256(summary)}  summary.json\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    weight, weight_words = _load_weight(args.model)
    with np.load(args.evidence) as evidence:
        payload = derive_audit(
            weight=weight,
            weight_words=weight_words,
            prefix_inputs=evidence["prefix_inputs"],
            current_input=evidence["current_input"],
            roundlock_pairs={
                "atlas_plus_one_page_down": (
                    evidence["candidate_down"],
                    evidence["native_down"],
                ),
                "post_residual_pre_norm": (
                    evidence["candidate_pre_norm"],
                    evidence["native_pre_norm"],
                ),
                "post_rms_norm_hidden": (
                    evidence["candidate_hidden"],
                    evidence["native_hidden"],
                ),
            },
        )
    payload["source"] = {
        "model_path": args.model.as_posix(),
        "evidence_path": args.evidence.as_posix(),
        "tensor_name": TENSOR_NAME,
        "model_sha256": _sha256(args.model),
        "evidence_sha256": _sha256(args.evidence),
    }
    if args.output_dir is not None:
        write_result(args.output_dir, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
