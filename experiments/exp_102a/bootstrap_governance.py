from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START = "<!-- REALITY-FIRST-EXECUTION:START -->"
END = "<!-- REALITY-FIRST-EXECUTION:END -->"

BLOCK = f"""{START}
## Reality-first authoritative execution

Every new core Gate has one authoritative arm: `REAL_EXECUTOR_ONLY`.
Future target tokens or hidden states, perfect selectors, free `N/A`, free transforms,
free metadata/workspace, free repair/fallback, unmeasured compression, and peak
throughput presented as sustained throughput are forbidden from satisfying a
promotion threshold. Synthetic or target-seeing calculations may appear only as
non-authoritative debugging diagnostics.

The authoritative arm must execute a finite-word causal path and charge candidate
generation, every target position, verification, mismatch repair, rollback,
fallback, transforms, packing, metadata, storage/host/device bytes, KV/cache,
workspace, fragmentation, and measured wall time. Missing quantities remain
`NOT TESTED`; they are never replaced by an ideal grant.

Normative detail: [`docs/research/REALITY_FIRST_EXECUTION_CONTRACT.md`](docs/research/REALITY_FIRST_EXECUTION_CONTRACT.md).
{END}"""


def replace_or_append(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if START in text or END in text:
        if text.count(START) != 1 or text.count(END) != 1:
            raise RuntimeError(f"malformed reality-first marker in {path}")
        prefix, rest = text.split(START, 1)
        _, suffix = rest.split(END, 1)
        updated = prefix.rstrip() + "\n\n" + BLOCK + suffix
    else:
        updated = text.rstrip() + "\n\n" + BLOCK + "\n"
    path.write_text(updated, encoding="utf-8")


def update_readme(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    old = """### Active frontier

The active stacked research PR is:

```text
PR #111
research/exp-101a-structured-direct-sum-gate
EXP-101A structured direct-sum composition Gate
```

EXP-101A tests an exact structured tensor source outside the prior AlphaTensor catalog. Until its authoritative result commit exists, it is an active Gate, not a scientific result.

"""
    new = """### Active frontier

EXP-101A was closed as `DIAGNOSTIC_ONLY`: its multiplication-only Gate granted
future blocks and zero-cost runtime components, so it cannot establish deployable
progress under the reality-first contract.

The replacement source branch is:

```text
research/exp-102a-causal-segment-delta-reality-gate
EXP-102A reality-first causal draft/verify Gate
```

EXP-102A executes real public draft and target checkpoints, charges every online
candidate/verification/repair/rebuild/state cost, uses no compression credit for
promotion, and requires exact terminal KV equality. Until a remote result commit
exists, it is an active Gate, not a scientific result.

"""
    if old in text:
        text = text.replace(old, new, 1)
    elif "EXP-102A reality-first causal draft/verify Gate" not in text:
        raise RuntimeError("README active-frontier block changed unexpectedly")
    path.write_text(text, encoding="utf-8")
    if START in text or "## Reality-first authoritative execution" not in text:
        replace_or_append(path)


def main() -> None:
    for relative in (
        "MISSION_AND_WORKING_PRINCIPLES.md",
        "AGENTS.md",
        "docs/REPOSITORY_COMMIT_AND_HANDOFF_MANDATE.md",
        "CREATIVE_RESEARCH_MANDATE.md",
    ):
        replace_or_append(ROOT / relative)
    update_readme(ROOT / "README.md")
    print("REALITY_FIRST_GOVERNANCE_APPLIED")


if __name__ == "__main__":
    main()
