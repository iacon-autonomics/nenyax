import subprocess
import sys

import pytest

from nenyax.plugins import KINDS, installed
from nenyax.scaffold import scaffold


def test_every_builtin_kind_is_listed():
    kinds = {p.kind for p in installed()}
    assert {"driver", "sandbox", "provider", "hub", "learner", "judge", "telemetry"} <= kinds
    assert set(KINDS) == {"driver", "sandbox", "provider", "hub", "learner", "judge", "telemetry"}


@pytest.mark.parametrize("kind", ["driver", "sandbox", "learner", "judge", "hub", "telemetry"])
def test_scaffolded_plugin_passes_its_own_conformance_test(kind, tmp_path, monkeypatch):
    """The generated package works and its generated test passes (CI also pip-installs it)."""
    import importlib.util

    import nenyax

    root = scaffold(kind, f"demo {kind}", tmp_path)
    pkg = next((root / "src").iterdir())
    monkeypatch.syspath_prepend(str(pkg.parent))
    module = importlib.import_module(pkg.name)
    if kind == "driver":  # stands in for the entry point that `pip install` would register
        nenyax.register(next(v for k, v in vars(module).items() if k.endswith("Driver"))())
    spec = importlib.util.spec_from_file_location(
        f"generated_{kind}", root / "tests" / "test_conformance.py"
    )
    generated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generated)
    for name, fn in vars(generated).items():
        if name.startswith("test_"):
            fn()


def test_registry_file_is_valid():
    proc = subprocess.run(
        [sys.executable, "tools/validate_registry.py"], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stdout
