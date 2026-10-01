"""Stack plugs on the runner: storage, sandboxed execution, tracking and run artifacts."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from nenyax import remote, sandboxed, storage


class FakeEvents:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []
        self.cancelled = False
        self.client = None
        self.run_id = 1

    def emit(self, type, payload, flush=False):
        self.events.append((type, payload))

    def flush(self):
        pass


ENV_PY = """
import nenyax

env = nenyax.define(
    "always-four",
    tasks=[{"prompt": f"What is {a} + {4 - a}?", "answer": "4"} for a in range(4)],
    score=lambda t, r: float(r.strip().endswith(t["answer"])),
    reference=lambda t: "Answer: 4",
)
"""
MODEL_PY = """
from nenyax.policy import FunctionModel

def make(**_):
    return FunctionModel(lambda messages: "Answer: 4")
"""


@pytest.fixture
def env_dir(tmp_path: Path) -> Path:
    root = tmp_path / "always-four"
    root.mkdir()
    (root / "env.py").write_text(ENV_PY)
    (root / "model.py").write_text(MODEL_PY)
    (root / "nenyax.toml").write_text(
        '[environment]\nname = "always-four"\nentrypoint = "env:env"\n'
    )
    return root


# -- storage -------------------------------------------------------------------------------------


def test_s3_splits_the_platform_key_and_uploads(tmp_path, monkeypatch):
    calls = {}

    class Client:
        def upload_file(self, path, bucket, key):
            calls["upload"] = (Path(path).read_text(), bucket, key)

    fake = types.SimpleNamespace(client=lambda name, **kw: calls.setdefault("kw", kw) and Client())
    monkeypatch.setitem(sys.modules, "boto3", fake)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "AKIDEXAMPLE:s3cret")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    f = tmp_path / "rows.jsonl"
    f.write_text("{}\n")
    store = storage.from_stack({"storage": {"plug": "s3", "config": {"uri": "s3://bucket/runs"}}})
    stored = store.put(f, "rows.jsonl", "dataset")
    assert stored.uri == "s3://bucket/runs/rows.jsonl" and stored.bytes == 3
    assert calls["upload"] == ("{}\n", "bucket", "runs/rows.jsonl")
    assert calls["kw"]["aws_access_key_id"] == "AKIDEXAMPLE"
    assert calls["kw"]["aws_secret_access_key"] == "s3cret"


def test_hf_hub_uploads_into_a_private_repo(tmp_path, monkeypatch):
    seen = {}

    class Api:
        def __init__(self, token=None):
            seen["token"] = token

        def create_repo(self, repo, **kw):
            seen["repo"] = (repo, kw)

        def upload_file(self, **kw):
            seen["upload"] = kw

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=Api))
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    f = tmp_path / "p.json"
    f.write_text("[]")
    store = storage.from_stack({"storage": {"plug": "hf", "config": {"repo": "me/runs"}}})
    assert store.put(f, "p.json").uri == "hf://datasets/me/runs/p.json"
    assert seen["token"] == "hf_x" and seen["repo"][1]["private"]
    assert seen["upload"]["path_in_repo"] == "p.json"


def test_storage_plug_needs_a_location_and_defaults_to_the_platform():
    with pytest.raises(storage.StorageError, match="location"):
        storage.from_stack({"storage": {"plug": "s3"}})
    assert storage.from_stack(None) is None  # no platform connection: nothing to store into
    plat = storage.from_stack(None, url="http://p", token="t", run_id=3)
    assert isinstance(plat, storage.NenyaxStorage)


# -- requirements --------------------------------------------------------------------------------


def test_requirements_fail_early_with_a_clear_reason():
    with pytest.raises(RuntimeError, match="needs docker.*'e2b' sandbox"):
        sandboxed.check_requirements(["docker"], "e2b")
    with pytest.raises(RuntimeError, match="this runner"):
        sandboxed.check_requirements(["docker"], "nenyax", docker_here=False)
    sandboxed.check_requirements(["network", "api_key"], "modal")  # fine


# -- the runner: artifacts and sandboxed execution -----------------------------------------------


def test_baseline_writes_a_trajectory_dataset(env_dir, tmp_path):
    out = remote.execute(
        "baseline",
        {"env": {"uri": str(env_dir)}, "params": {"n": 4}},
        FakeEvents(),
        outdir=tmp_path / "out",
    )
    assert out["score"] == 1.0 and out["episodes"] == 4
    names = {Path(a["path"]).name for a in out["_artifacts"]}
    assert "trajectories-raw.jsonl" in names
    raw = [
        json.loads(x)
        for x in (tmp_path / "out" / "trajectories-raw.jsonl").read_text().splitlines()
    ]
    assert len(raw) == 4 and all(r["score"] == 1.0 for r in raw)


def test_training_saves_the_learned_prompt_and_datasets(env_dir, tmp_path):
    config = {
        "env": {"uri": str(env_dir)},
        "model": {"use": "model:make"},
        "learner": {"use": "incontext"},
        "params": {"rounds": 1, "batch": 2, "group_size": 2},
    }
    out = remote.execute("train", config, FakeEvents(), outdir=tmp_path / "out")
    kinds = {a["kind"] for a in out["_artifacts"]}
    assert {"dataset", "prompt"} <= kinds
    prompt = next(a for a in out["_artifacts"] if a["kind"] == "prompt")
    shots = json.loads(Path(prompt["path"]).read_text())
    assert shots and shots[0]["messages"][-1]["content"] == "Answer: 4"


def test_unknown_tracking_plug_is_skipped_not_fatal():
    assert remote._callbacks({"stack": {"tracking": [{"plug": "no-such-tracker"}]}}) == []


def test_runs_inside_the_process_sandbox_and_streams_events(env_dir):
    events = FakeEvents()
    config = {
        "env": {"uri": str(env_dir)},
        "params": {"n": 3},
        "stack": {"sandbox": {"plug": "process"}},
    }
    out = sandboxed.execute_in_sandbox("baseline", config, events)
    assert out["score"] == 1.0 and out["episodes"] == 3
    episodes = [p for t, p in events.events if t == "episode"]
    assert len(episodes) == 3  # forwarded from the sandbox while it ran
    local = [Path(a["path"]) for a in out["_artifacts"]]
    assert local and all(p.exists() for p in local)  # copied out of the sandbox


def test_publish_stores_each_file_with_the_plug(tmp_path):
    f = tmp_path / "a.jsonl"
    f.write_text("x\n")

    class Mem(storage.Storage):
        plug = "mem"

        def __init__(self):
            self.put_calls = []

        def put(self, local, name, kind="other"):
            self.put_calls.append((name, kind))
            return storage.Stored(uri=f"mem://{name}", bytes=2, sha256="0" * 64)

    store = Mem()
    out = storage.publish(
        [{"path": str(f), "name": "a.jsonl", "kind": "dataset"}],
        store,
        url=None,
        token=None,
        run_id=None,
    )
    assert store.put_calls == [("a.jsonl", "dataset")] and out[0]["uri"] == "mem://a.jsonl"


def test_two_env_folders_with_the_same_module_name_load_their_own_env(tmp_path):
    from nenyax.registry import load

    for name in ("first", "second"):
        root = tmp_path / name
        root.mkdir()
        (root / "env.py").write_text(
            f"import nenyax\nenv = nenyax.define({name!r}, tasks=[{{'prompt': 'x'}}], "
            "score=lambda t, r: 0.0)\n"
        )
        toml = f'[environment]\nname = "{name}"\nentrypoint = "env:env"\n'
        (root / "nenyax.toml").write_text(toml)
    assert load(str(tmp_path / "first")).manifest.name == "first"
    assert load(str(tmp_path / "second")).manifest.name == "second"
