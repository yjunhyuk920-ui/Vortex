"""Check exact evidence bytes, Git clean filters, and optionally the index."""
import argparse
import hashlib
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true")
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    repo = package.parents[1]
    checked = 0
    for line in (package / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = package / relative
        assert path.resolve().is_relative_to(package), relative
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected, relative
        git_path = path.relative_to(repo).as_posix()
        before = subprocess.check_output(["git", "hash-object", "--no-filters", "--", git_path], cwd=repo)
        after = subprocess.check_output(["git", "hash-object", "--path=" + git_path, "--", git_path], cwd=repo)
        assert before == after, "Git clean changes " + relative
        if args.index:
            indexed = subprocess.check_output(["git", "show", ":" + git_path], cwd=repo)
            assert indexed == raw, "Index bytes differ: " + relative
        checked += 1
    if args.index:
        manifest_path = (package / "CHECKSUMS.sha256").relative_to(repo).as_posix()
        assert subprocess.check_output(["git", "show", ":" + manifest_path], cwd=repo) == (package / "CHECKSUMS.sha256").read_bytes()
    print(f"PASS: {checked} manifest entries; Git clean preserves bytes; index_checked={args.index}")


if __name__ == "__main__":
    main()
