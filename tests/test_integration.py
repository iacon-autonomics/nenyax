"""Integration tests against real third-party environment packages.

Each test is skipped unless its package (and, for some, a service) is available:

* ``NENYAX_NETWORK=1``       enables tests against public Hugging Face Spaces
* ``NENYAX_NEMO_GYM_URL``    a running NeMo Gym ``reasoning_gym`` resources server
* ``NENYAX_NEMO_GYM_DATA``   its example JSONL
* Docker must be running for Harbor tests
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

import nenyax
from nenyax import FunctionModel, Message, NativeAgent, Tier, ToolCall, check

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"

pytestmark = pytest.mark.integration


def needs(module: str):
    return pytest.mark.skipif(find_spec(module) is None, reason=f"{module} not installed")


def docker_running() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


# -- Gymnasium ----------------------------------------------------------------------------------


@needs("gymnasium")
def test_gymnasium_cartpole_is_trainable_but_has_no_oracle():
    env = nenyax.load("gymnasium:CartPole-v1")
    report = check(env)
    assert report.tier == Tier.TRAINABLE
    traj = env.rollout(env.random_policy(seed=0), seed=0)
    assert traj.ok and len(traj.steps) > 1 and traj.score == len(traj.steps)


# -- reasoning-gym ------------------------------------------------------------------------------


@needs("reasoning_gym")
@pytest.mark.parametrize("name", ["basic_arithmetic", "leg_counting", "knights_knaves"])
def test_reasoning_gym_datasets_are_audited(name):
    env = nenyax.load(f"reasoning_gym:{name}?size=5&seed=1")
    assert check(env).tier == Tier.AUDITED


# -- Verifiers ----------------------------------------------------------------------------------


@pytest.fixture
def verifiers_examples(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES / "verifiers"))


def _solve(messages):
    expr = messages[-1].content.removeprefix("What is ").rstrip("?")
    return f"<answer>{eval(expr)}</answer>"  # the toy questions are arithmetic literals


@needs("verifiers")
def test_verifiers_legacy_env_is_audited(verifiers_examples):
    env = nenyax.load(
        "verifiers:toy_math_legacy",
        reference="<answer>{answer}</answer>",
        null="<answer>?</answer>",
    )
    assert env.manifest.source_format == "verifiers/legacy"
    assert check(env).tier == Tier.AUDITED
    traj = env.rollout(FunctionModel(_solve), task=env.tasks()[1])
    assert traj.score == 1.0 and len(traj.model_calls) == 1


@needs("verifiers")
def test_verifiers_v1_taskset_is_audited(verifiers_examples):
    env = nenyax.load(
        "verifiers:toy_math_v1", reference="<answer>{answer}</answer>", null="<answer>?</answer>"
    )
    assert env.manifest.source_format == "verifiers/v1"
    assert check(env, samples=2).tier == Tier.AUDITED


@needs("verifiers")
def test_verifiers_tool_env_round_trips_openai_tool_calls(verifiers_examples):
    import toy_math_legacy

    from nenyax.formats.verifiers import VerifiersEnvironment

    env = VerifiersEnvironment(toy_math_legacy.load_tool_environment())

    def use_tool(messages):
        if messages[-1].role == "tool":
            return f"<answer>{messages[-1].content}</answer>"
        call = ToolCall(id="c1", name="add", arguments={"a": 17, "b": 25})
        return Message(role="assistant", tool_calls=[call])

    traj = env.rollout(FunctionModel(use_tool))
    assert traj.score == 1.0
    assert traj.judgment.components["add_calls"] == 1.0
    assert [m.role for m in traj.messages] == ["user", "assistant", "tool", "assistant"]


@pytest.mark.skipif(find_spec("gsm8k") is None, reason="run: vf-install primeintellect/gsm8k")
def test_verifiers_hub_gsm8k_scores_gold_and_wrong_answers():
    env = nenyax.load("verifiers:gsm8k")
    task = env.tasks(limit=1)[0]
    gold = str(env._fields(0)["answer"]).split("####")[-1].strip()
    right = env.rollout(FunctionModel(lambda m: f"\\boxed{{{gold}}}"), task=task)
    wrong = env.rollout(FunctionModel(lambda m: "\\boxed{-1}"), task=task)
    assert (right.score, wrong.score) == (1.0, 0.0)


# -- Harbor -------------------------------------------------------------------------------------


@needs("harbor")
@pytest.mark.docker
@pytest.mark.skipif(not docker_running(), reason="Docker is not running")
def test_harbor_task_is_audited_and_drivable_by_a_chat_model(tmp_path):
    env = nenyax.load(f"harbor:{EXAMPLES / 'harbor' / 'hello'}", trials_dir=str(tmp_path))
    assert env.rollout(NativeAgent("oracle")).score == 1.0
    assert env.rollout(NativeAgent("nop")).score == 0.0

    def terminus(messages):
        done = any(m.role == "assistant" for m in messages)
        cmds = [] if done else [{"keystrokes": "echo hi > /app/hello.txt\n", "duration": 0.5}]
        reply = {"analysis": "", "plan": "", "commands": cmds, "task_complete": done}
        return json.dumps(reply)

    traj = env.rollout(FunctionModel(terminus))
    assert traj.score == 1.0 and len(traj.model_calls) >= 2
    assert check(env, samples=1).tier == Tier.AUDITED


# -- OpenEnv ------------------------------------------------------------------------------------


@needs("openenv")
@pytest.mark.skipif(os.environ.get("NENYAX_NETWORK") != "1", reason="set NENYAX_NETWORK=1")
def test_openenv_public_space_is_evaluable():
    env = nenyax.load("openenv:https://openenv-openspiel-env.hf.space")

    def first_legal(obs):
        return {"action_id": obs.data["legal_actions"][0]}

    report = check(env, probe=first_legal, samples=1)
    assert report.tier == Tier.EVALUABLE  # the Space does not declare seeded reset


@needs("openenv")
@needs("reasoning_gym")
def test_any_step_env_exports_to_openenv_and_loads_back(unused_tcp_port):
    """Port in (reasoning-gym) -> port out (OpenEnv server) -> port back in (OpenEnv driver)."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "nenyax.cli",
            "serve",
            "reasoning_gym:leg_counting?size=5",
            "--port",
            str(unused_tcp_port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        url = f"http://127.0.0.1:{unused_tcp_port}"
        _wait_for(url + "/health")
        env = nenyax.load(f"openenv:{url}", seeded=True)
        assert env.manifest.action_schema["required"] == ["value"]
        traj = env.rollout(lambda obs: {"value": "<answer>0</answer>"}, seed=0)
        judgment = traj.steps[-1].observation.info["metadata"]["nenyax_judgment"]
        assert traj.ok and judgment["judge"].startswith("reasoning_gym:leg_counting")
        assert check(env, probe=lambda obs: {"value": "x"}).tier == Tier.TRAINABLE
    finally:
        proc.terminate()
        proc.wait(timeout=10)


# -- NeMo Gym -----------------------------------------------------------------------------------


@pytest.mark.skipif("NENYAX_NEMO_GYM_URL" not in os.environ, reason="set NENYAX_NEMO_GYM_URL")
def test_nemo_gym_resources_server_is_audited():
    env = nenyax.load(
        f"nemo_gym:{os.environ['NENYAX_NEMO_GYM_URL']}",
        data=os.environ["NENYAX_NEMO_GYM_DATA"],
        reference="<answer>{answer}</answer>",
        null="<answer>no idea</answer>",
    )
    assert check(env).tier == Tier.AUDITED


# -- helpers ------------------------------------------------------------------------------------


@pytest.fixture
def unused_tcp_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for(url: str, timeout_s: float = 30.0) -> None:
    import time
    import urllib.request

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(url)
