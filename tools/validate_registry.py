"""Validate registry/plugins.toml (run before merging any PR that touches it).

python tools/validate_registry.py              # schema checks only
python tools/validate_registry.py --install    # also pip-install each plugin and check that
                                               # its entry point loads under the right group
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from importlib.metadata import entry_points
from pathlib import Path

from nenyax.plugins import KINDS

REQUIRED = ("name", "kind", "package", "maintainer", "repo", "license", "open_source", "test")


def validate(path: Path) -> list[str]:
    errors, seen = [], set()
    entries = tomllib.loads(path.read_text()).get("plugin", [])
    for i, p in enumerate(entries):
        where = f"plugin #{i + 1} ({p.get('name', '?')})"
        errors += [f"{where}: missing '{f}'" for f in REQUIRED if f not in p]
        if p.get("kind") not in KINDS:
            errors.append(f"{where}: kind must be one of {sorted(KINDS)}")
        key = (p.get("kind"), p.get("name"))
        if key in seen:
            errors.append(f"{where}: duplicate {key}")
        seen.add(key)
        if not str(p.get("repo", "")).startswith("https://"):
            errors.append(f"{where}: repo must be an https URL")
    return errors


def install_and_load(path: Path) -> list[str]:
    errors = []
    for p in tomllib.loads(path.read_text()).get("plugin", []):
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", p["package"]],
            capture_output=True,
            text=True,
        )
        if proc.returncode:
            errors.append(f"{p['name']}: pip install {p['package']} failed: {proc.stderr[-300:]}")
            continue
        group = KINDS[p["kind"]].group
        names = {ep.name for ep in entry_points(group=group)}
        if p["name"] not in names:
            errors.append(f"{p['name']}: no entry point {p['name']!r} in group {group}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true")
    parser.add_argument("path", nargs="?", default="registry/plugins.toml")
    args = parser.parse_args()
    errors = validate(Path(args.path))
    if not errors and args.install:
        errors = install_and_load(Path(args.path))
    for e in errors:
        print("✘", e)
    print("registry OK" if not errors else f"{len(errors)} problem(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
