"""List CI's standalone Python test files or pytest targets, one per line."""
import argparse
import fnmatch
import os
from pathlib import Path
import sys


SEMANTIC_DIR = "tests/semantic"
TEST_GLOB = "test_*.py"
PYTEST_TARGETS = "tests/pytest-targets.txt"


def python_test_files(root):
    """Find the non-semantic Python tests without following symlinks."""
    root = Path(root)
    files = []
    for base, dirs, names in os.walk(root / "tests"):
        folder = Path(base)
        dirs[:] = [d for d in dirs if (folder / d).relative_to(root).as_posix() != SEMANTIC_DIR]
        files.extend((folder / name).relative_to(root).as_posix()
                     for name in names if fnmatch.fnmatchcase(name, TEST_GLOB)
                     and not (folder / name).is_symlink() and (folder / name).is_file())
    return sorted(files)


def routes_for(root):
    """Return (standalone files, pytest targets); only whole-file entries change the default route."""
    files = python_test_files(root)
    known = set(files)
    targets, modules, nodes = set(), set(), set()
    with (Path(root) / PYTEST_TARGETS).open(encoding="utf-8") as manifest:
        for number, line in enumerate(manifest, 1):
            target = line.strip()
            if not target or target.startswith("#"):
                continue
            path, separator, node = target.partition("::")
            if path not in known or (separator and not node):
                raise ValueError(f"{PYTEST_TARGETS}:{number}: invalid or missing test target: {target}")
            if target in targets or path in modules or (not separator and path in nodes):
                raise ValueError(f"{PYTEST_TARGETS}:{number}: duplicate or overlapping test target: {target}")
            targets.add(target)
            (nodes if separator else modules).add(path)
    return [path for path in files if path not in modules], sorted(targets)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner", choices=("stdlib", "pytest"))
    args = parser.parse_args(argv)
    try:
        stdlib, pytest = routes_for(Path(__file__).resolve().parent.parent)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    selected = stdlib if args.runner == "stdlib" else pytest
    if not selected:
        parser.error(f"no {args.runner} tests selected")
    # Bash reads the paths from a pipe and would keep the CR in Windows' default CRLF output.
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    for target in selected:
        print(target)


if __name__ == "__main__":
    main()
