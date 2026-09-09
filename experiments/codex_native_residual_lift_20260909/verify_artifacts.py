"""Verify stored evidence; does not rerun native calls or prove the theorem."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    manifest = read("checksums.json")
    for name, digest in manifest.items():
        require(sha(HERE / name) == digest, "manifest mismatch: " + name)
    ordered = read("results.json")
    native = read("torch_cpu_results.json")
    integer = read("integer_audit.json")
    admission = read("admission_validation.json")
    for record, source in ((ordered, "lift.py"), (native, "torch_cpu_audit.py"),
                           (integer, "integer_audit.py"), (admission, "lift.py")):
        require(record["source_sha256"] == sha(HERE / source), "executed source: " + source)
    require(ordered["preregistration_sha256"] == sha(HERE / "PREREGISTRATION.md"), "preregistration hash")
    require(ordered["case_count"] == len(ordered["cases"]) == 106, "ordered count")
    for row in ordered["cases"]:
        require(row["decoded_counts"] == row["truth_counts"], "stored decoded count")
        require(row["parity"] == [k % 2 for k in row["truth_counts"]], "stored parity")
        require(row["b"] == row["n"].bit_length(), "bias width")
        require(row["D"] == row["n"] + row["m"] * row["b"], "augmented width")
        require(row["numeric_accounting"]["two_matvec_products"] == 2 * row["m"] * row["D"], "dense products")
    require(native["operator_case_count"] == len(native["rows"]) == 12, "native case count")
    require(native["native_call_count"] == 2 * len(native["rows"]) == 24, "native call count")
    require(not native["cuda_executed"] and not native["mismatch_cases"], "native scope/status")
    for row in native["rows"]:
        for index in (0, 1):
            require(row[f"y{index}_words"] == row[f"expected_y{index}_words"], "native WORD mismatch")
        require(row["decoded"] == row["counts"] and row["rng_preserved"], "native decode/RNG")
    require(integer["first_cases"] == 128 * 16385, "integer count pairs")
    require(integer["second_cases"] == 128 * 131, "integer residual pairs")
    require(len(admission["refusals"]) == 9 and all(r["refused"] for r in admission["refusals"]), "domain refusals")
    require(admission["negative_zero_decoder"] == 0, "negative zero decoder")
    print(json.dumps({"status": "STORED_EVIDENCE_VERIFIED", "manifest_files": len(manifest),
                      "ordered_cases": 106, "native_word_cases": 12,
                      "native_calls_rerun": 0, "mission_status": "UNCHANGED_OPEN"}, sort_keys=True))


if __name__ == "__main__":
    main()
