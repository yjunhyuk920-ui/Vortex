"""Validate the CTC policy's formatting and guard clauses, NOT the research theory.
Run from a complete repo root: python docs/governance/validate_ctc_20260905.py
For a local payload set pass --root PATH. No network or external dependencies.
"""
from __future__ import annotations
import argparse, hashlib, json, posixpath, re
from pathlib import Path

POLICIES = (
    'AGENTS.md', 'MISSION_AND_WORKING_PRINCIPLES.md', 'CREATIVE_RESEARCH_MANDATE.md',
    'docs/CONSTRUCTIVE_THEORY_CONTRACT.md', 'docs/PROOF_FIRST_CONTRACT.md',
    'docs/RESEARCH_EFFICIENCY_CONTRACT.md', 'docs/WORK_SESSION_PROTOCOL.md',
    'docs/REPOSITORY_COMMIT_AND_HANDOFF_MANDATE.md',
    'docs/governance/CTC_20260905_DECISION.md', 'NEXT_EXPERIMENT.md',
)

def validate(root: Path) -> dict:
    errors: list[str] = []
    texts: dict[str, str] = {}
    hashes: dict[str, str] = {}
    links = 0
    for name in POLICIES:
        try:
            raw = (root / name).read_bytes()
            text = raw.decode('utf-8')
        except (OSError, UnicodeError) as exc:
            errors.append(f'{name}: {exc}')
            continue
        texts[name] = text
        hashes[name] = hashlib.sha256(raw).hexdigest()
        if not text.endswith('\n') or '\r' in text or '\0' in text:
            errors.append(f'{name}: invalid newline/NUL')
        if 'CTC-2026-09-05' not in text:
            errors.append(f'{name}: missing policy version')
        if len(re.findall(r'^```', text, re.M)) % 2:
            errors.append(f'{name}: unbalanced fence')
        if any(line.rstrip() != line for line in text.splitlines()):
            errors.append(f'{name}: trailing whitespace')
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
            if '://' in target or target.startswith('#'):
                continue
            dest = posixpath.normpath(posixpath.join(posixpath.dirname(name), target.split('#')[0]))
            links += 1
            if not (root / dest).is_file():
                errors.append(f'{name}: missing link {dest}')
    guards = {
        'docs/CONSTRUCTIVE_THEORY_CONTRACT.md': (
            'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'sufficient',
            'baseline lower bound', 'essential algorithmic assumptions',
            'THEORY_STATUS=NOT_ESTABLISHED|CONDITIONAL|VERIFIED_IN_MODEL',
            'HARDWARE_STATUS=NOT_TESTED|PARTIAL|E7_VERIFIED',
            'HANDOFF_STATUS=IN_PROGRESS|REMOTE_COMMIT_VERIFIED|BLOCKED_REMOTE_WRITE',
            'A >> 1', 'not an automatic stopping criterion',
        ),
        'AGENTS.md': ('405B', '<=8 GiB', 'p50 <=1.2x', 'p95 <=1.5x',
            'GITHUB_ACTIONS_REQUIRED=false', 'GITHUB_REEXECUTION_REQUIRED=false',
            'REAL_EXECUTOR_ONLY', 'Never force-push', 'three materially different'),
        'docs/WORK_SESSION_PROTOCOL.md': ('Documentation-only validation',
            'Governance-only', 'hypotheses', 'REMOTE_COMMIT_VERIFIED'),
    }
    # Case-insensitive containment only; these guards do not prove semantics.
    for name, required in guards.items():
        text = texts.get(name, '').lower()
        for token in required:
            if token.lower() not in text:
                errors.append(f'{name}: missing guard {token}')
    return {'version': 'CTC-2026-09-05', 'validation_scope': 'policy_format_links_guard_clauses_only',
        'status': 'PASS' if not errors else 'FAIL', 'policy_files': len(texts),
        'local_markdown_links_checked': links, 'errors': errors,
        'sha256': hashes, 'proves_research_theory': False,
        'runtime_tests_run': False, 'target_hardware_tests_run': False}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = validate(args.root.resolve())
    output = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    print(output, end='')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding='utf-8')
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
