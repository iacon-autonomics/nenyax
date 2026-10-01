"""The platform CLI: every command parses, packing is deterministic, local folders load."""

from __future__ import annotations

import argparse
import io
import tarfile

import pytest

from nenyax import platform, platform_cli
from nenyax.registry import load


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nenyax")
    platform_cli.register(parser.add_subparsers(dest="command"))
    return parser


@pytest.mark.parametrize(
    "argv",
    [
        ["auth", "login", "--no-browser"],
        ["auth", "status", "--json"],
        ["env", "list", "sudoku", "--kind", "environment"],
        ["env", "view", "dev/gsm8k-mini", "--readme"],
        ["env", "push", ".", "--public"],
        ["env", "try", "reasoning-gym/basic_arithmetic", "--model", "openai/gpt-4o-mini", "-w"],
        ["project", "create", "arith", "--env", "reasoning-gym/basic_arithmetic", "--runners"],
        ["project", "edit", "arith", "--pipeline", "baseline,train,check", "--compute", "cloud"],
        ["project", "run", "arith", "baseline", "-p", "n=5", "--watch"],
        ["project", "export", "arith", "-o", "x.toml"],
        ["runs", "watch", "12"],
        ["connection", "add", "openai", "--key", "-"],
        ["runners", "add", "lab-4090"],
        ["org", "add-member", "iacon", "ada", "--role", "admin"],
        ["api", "GET", "/projects"],
    ],
)
def test_every_command_parses(argv):
    args = _parser().parse_args(argv)
    assert callable(args.func)


def test_params_are_typed():
    parsed = platform_cli._params(["n=5", "name=x", "flag=true"])
    assert parsed == {"n": 5, "name": "x", "flag": True}


def test_new_env_loads_by_folder_and_packs_deterministically(tmp_path):
    root = platform.new_env("demo-env", tmp_path)
    env = load(str(root))
    assert env.tasks() and env.manifest.name == "demo-env"
    first, second = platform.pack(root), platform.pack(root)
    assert first == second  # no timestamps: the same files are the same bytes
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as tar:
        assert sorted(tar.getnames()) == ["README.md", "env.py", "nenyax.toml"]


def test_manifest_needs_name_and_entrypoint(tmp_path):
    (tmp_path / "nenyax.toml").write_text('[environment]\nname = "x"\n')
    with pytest.raises(platform.PlatformError):
        platform.read_manifest(tmp_path)
