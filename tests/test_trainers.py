"""External trainers: generated jobs, and rewards that come from the environment's own verifier."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tomllib
import urllib.request
from importlib.util import find_spec
from pathlib import Path

import pytest

import nenyax
from nenyax import FunctionModel
from nenyax.config import build
from nenyax.train import train
from nenyax.trainers import ExternalLearner, TrainerUnavailable, catalog, get
from nenyax.trainers.base import Job, JobResult, Trainer
from nenyax.trainers.bridge import Bridge, completion_text, export_tasks

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
RGYM = "reasoning_gym:basic_arithmetic?size=8"
needs_verifiers = pytest.mark.skipif(find_spec("verifiers") is None, reason="needs verifiers")


def _quiz():
    tasks = [{"prompt": f"What is {a}+{a}?", "answer": str(2 * a)} for a in range(1, 9)]
    return nenyax.define(
        "quiz",
        tasks=tasks,
        score=lambda t, r: float(r.strip().split()[-1:] == [t["answer"]]),
        reference=lambda t: t["answer"],
    )


def _gold(env, task):
    """The correct reply for a task, from the environment's own reference policy."""
    with env.session(task=task) as s:
        return env.reference_policy(task)(s.observation)


def _import(path: Path, name: str):
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(path.parent))
        sys.modules.pop("nenyax_bridge", None)


# -- the catalog and the bridge -------------------------------------------------------------------


def test_catalog_is_honest_about_what_runs():
    entries = {c["id"]: c for c in catalog()}
    assert set(entries) == {"nenyax", "trl", "unsloth", "verl", "prime-rl", "openrlhf", "tinker"}
    assert entries["nenyax"]["status"] == "ready"
    assert all(e["status"] == "beta" for k, e in entries.items() if k != "nenyax")
    for e in entries.values():
        assert e["config_schema"]["type"] == "object" and "requirements" in e
    assert "1 GPU" in entries["trl"]["needs"] and "TINKER_API_KEY" in entries["tinker"]["needs"]


def test_bridge_scores_real_completions_with_the_real_verifier():
    env = nenyax.load(RGYM)
    task = env.tasks()[0]
    bridge = Bridge(env)
    assert bridge.score(task.id, _gold(env, task)) == 1.0
    assert bridge.score(task.id, "<answer>-1</answer>") == 0.0
    # Trainers hand completions over in every shape; all mean the same reply.
    gold = _gold(env, task)
    for shape in (
        gold,
        {"role": "assistant", "content": gold},
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": gold}],
    ):
        assert bridge.score(task.id, shape) == 1.0
    quiz = Bridge(_quiz())
    assert (quiz.score("2", "6"), quiz.score("2", "7")) == (1.0, 0.0)


def test_generated_bridge_rebuilds_the_environment_in_another_process(tmp_path):
    job = get("trl").prepare(nenyax.load(RGYM), MODEL, {}, tmp_path)
    env = nenyax.load(RGYM)
    task = env.tasks()[0]
    gold, tid = _gold(env, task), task.id
    code = f"import nenyax_bridge as b; print(b.score({tid!r}, {gold!r}), b.score({tid!r}, 'no'))"
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=job.workdir, capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["1.0", "0.0"]


def test_export_needs_a_spec_external_processes_can_rebuild():
    env = _quiz()  # defined in Python: no URI to rebuild it from
    with pytest.raises(ValueError, match='config\\["env"\\]'):
        get("trl").prepare(env, MODEL, {}, Path("/tmp/never-written"))
    rows = export_tasks(env)
    assert rows[0] == {"task_id": "0", "prompt": "What is 1+1?", "answer": "2", "split": "train"}


# -- TRL and Unsloth --------------------------------------------------------------------------


def test_trl_grpo_job_trains_on_train_tasks_and_rewards_with_the_verifier(tmp_path):
    env = nenyax.load(RGYM)
    job = get("trl").prepare(env, MODEL, {"max_steps": 20, "num_generations": 4}, tmp_path)
    train_rows = [json.loads(x) for x in job.files["train_data"].read_text().splitlines()]
    eval_rows = [json.loads(x) for x in job.files["eval_data"].read_text().splitlines()]
    assert {r["task_id"] for r in train_rows}.isdisjoint({r["task_id"] for r in eval_rows})
    assert len(train_rows) + len(eval_rows) == 8
    script = job.files["script"].read_text()
    compile(script, "train_trl.py", "exec")
    assert "GRPOTrainer(" in script and "reward_funcs=[nenyax_reward]" in script
    assert job.command[-1] == "train_trl.py" and job.checkpoint_dir == tmp_path / "checkpoint"
    assert job.meta["args"]["max_steps"] == 20 and job.meta["args"]["num_generations"] == 4
    # The reward TRL will call: one score per completion, from the environment.
    module = _import(job.files["script"], "train_trl")
    task = env.tasks()[0]
    good = [{"role": "assistant", "content": _gold(env, task)}]
    bad = [{"role": "assistant", "content": "<answer>0</answer>"}]
    assert module.nenyax_reward(completions=[good, bad], task_id=[task.id, task.id]) == [1.0, 0.0]


def test_trl_dpo_needs_pairs(tmp_path):
    env = nenyax.load(RGYM)
    with pytest.raises(ValueError, match="pairs"):
        get("trl").prepare(env, MODEL, {"method": "dpo"}, tmp_path)
    job = get("trl").prepare(env, MODEL, {"method": "dpo", "pairs": "pairs.jsonl"}, tmp_path)
    assert "DPOTrainer(" in job.files["script"].read_text()
    assert job.meta["args"]["beta"] == 0.1


def test_unsloth_loads_with_unsloth(tmp_path):
    job = get("unsloth").prepare(nenyax.load(RGYM), MODEL, {"lora_rank": 32}, tmp_path)
    script = job.files["script"].read_text()
    assert "FastLanguageModel.from_pretrained" in script and "get_peft_model" in script
    compile(script, "train_trl.py", "exec")


@pytest.mark.skipif(find_spec("trl") is not None, reason="trl is installed here")
def test_run_says_exactly_what_is_missing_and_how_to_launch_elsewhere(tmp_path):
    job = get("trl").prepare(nenyax.load(RGYM), MODEL, {}, tmp_path)
    with pytest.raises(TrainerUnavailable) as e:
        get("trl").run(job)
    assert "needs trl" in str(e.value) and "train_trl.py" in str(e.value)


# -- verl --------------------------------------------------------------------------------------


def test_verl_parquet_and_reward_function(tmp_path):
    import pandas as pd

    env = nenyax.load(RGYM)
    job = get("verl").prepare(env, MODEL, {"rollout_n": 4, "gpus": 2}, tmp_path)
    df = pd.read_parquet(job.files["train_data"])
    assert set(df.columns) >= {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    row = df.iloc[0]
    assert row["prompt"][0]["role"] == "user" and row["extra_info"]["task_id"]
    over = job.meta["overrides"]
    assert over["custom_reward_function.path"] == str(job.files["reward"])
    assert over["actor_rollout_ref.rollout.n"] == 4 and over["trainer.n_gpus_per_node"] == 2
    assert job.command[1:3] == ["-m", "verl.trainer.main_ppo"]
    assert f"actor_rollout_ref.model.path={MODEL}" in job.command
    reward = _import(job.files["reward"], "nenyax_reward")
    task = next(t for t in env.tasks() if t.id == row["extra_info"]["task_id"])
    info = {"task_id": task.id}
    assert reward.compute_score("nenyax/x", _gold(env, task), "", info) == 1.0
    assert reward.compute_score("nenyax/x", "<answer>0</answer>", "", info) == 0.0


# -- prime-rl and the verifiers adapter --------------------------------------------------------


@needs_verifiers
def test_any_step_env_round_trips_through_verifiers():
    """Nenyax env -> verifiers env -> Nenyax's verifiers driver -> scored like the original."""
    from nenyax.formats.verifiers import VerifiersEnvironment
    from nenyax.trainers.prime_rl import to_verifiers

    quiz = _quiz()
    back = VerifiersEnvironment(to_verifiers(quiz))
    assert back.manifest.num_tasks == 8
    answers = {t.prompt: t.metadata["answer"] for t in quiz.tasks()}

    def solve(messages):
        return answers[messages[-1].content]

    assert back.rollout(FunctionModel(solve), task=back.tasks()[2]).score == 1.0
    assert back.rollout(FunctionModel(lambda m: "0"), task=back.tasks()[2]).score == 0.0


@needs_verifiers
def test_multi_turn_env_round_trips_through_verifiers():
    from nenyax.formats.verifiers import VerifiersEnvironment
    from nenyax.trainers.prime_rl import to_verifiers

    def step(state, action):
        state["turns"] += 1
        hit = str(action).strip() == str(state["secret"])
        return (1.0 if hit else 0.0), hit or state["turns"] >= 4

    guess = nenyax.define(
        "guess",
        reset=lambda seed: {"secret": 3, "turns": 0},
        observe=lambda s: f"Guess a digit. Turns so far: {s['turns']}",
        step=step,
    )
    guess.tasks = lambda split=None, limit=None: [nenyax.TaskRef(id="0", prompt="Guess a digit.")]
    back = VerifiersEnvironment(to_verifiers(guess, multi_turn=True))
    replies = iter(["1", "2", "3"])
    traj = back.rollout(FunctionModel(lambda m: next(replies)))
    assert traj.score == 1.0
    assert [m.role for m in traj.messages].count("assistant") == 3


@needs_verifiers
def test_prime_rl_job(tmp_path):
    job = get("prime-rl").prepare(nenyax.load(RGYM), MODEL, {"rollouts_per_example": 16}, tmp_path)
    config = tomllib.loads(job.files["config"].read_text())
    assert config["model"]["name"] == MODEL
    assert config["orchestrator"]["rollouts_per_example"] == 16
    assert config["orchestrator"]["environment"]["id"] == "nenyax_verifiers_env"
    assert job.command[:3] == ["uv", "run", "rl"]
    module = _import(job.files["environment"], "nenyax_verifiers_env")
    vf_env = module.load_environment()
    assert len(vf_env.get_eval_dataset()) == 8


# -- OpenRLHF ----------------------------------------------------------------------------------


def test_openrlhf_reward_server_scores_with_the_environment(tmp_path):
    from nenyax.trainers.openrlhf import RewardServer

    env = nenyax.load(RGYM)
    job = get("openrlhf").prepare(env, MODEL, {"port": 5099}, tmp_path)
    assert "--remote_rm_url" in job.command and "http://127.0.0.1:5099/get_reward" in job.command
    assert job.command[job.command.index("--label_key") + 1] == "task_id"
    assert job.sidecars == [[sys.executable, "reward_server.py"]]

    server = RewardServer(Bridge(env)).start()
    try:
        task = env.tasks()[0]
        prompt = "PROMPT:"
        body = {
            "query": [prompt + _gold(env, task), prompt + "<answer>0</answer>"],
            "prompts": [prompt, prompt],
            "labels": [task.id, task.id],
        }
        req = urllib.request.Request(
            server.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        reply = json.loads(urllib.request.urlopen(req, timeout=10).read())
        assert reply["rewards"] == [1.0, 0.0]
    finally:
        server.stop()


# -- trainers as learners, inside nenyax.train ----------------------------------------------------


class _ScriptTrainer(Trainer):
    """A real subprocess job that reports metrics the way generated trainer scripts do."""

    id, name = "script", "script trainer"

    def prepare(self, env, model, config, workdir):
        workdir.mkdir(parents=True, exist_ok=True)
        script = workdir / "fake_train.py"
        script.write_text(
            "import json, pathlib\n"
            "pathlib.Path('checkpoint').mkdir(exist_ok=True)\n"
            "print('NENYAX_METRIC ' + json.dumps({'step': 10, 'reward': 0.75}))\n"
        )
        return Job(
            self.id,
            workdir,
            [sys.executable, "fake_train.py"],
            checkpoint_dir=workdir / "checkpoint",
        )


def test_external_trainer_runs_inside_train_with_the_gate(tmp_path):
    env = _quiz()
    answers = {t.prompt: t.metadata["answer"] for t in env.tasks()}
    before = FunctionModel(lambda m: "0")  # the base model gets everything wrong
    after = FunctionModel(lambda m: answers[m[-1].content])  # the trained one gets it right
    learner = ExternalLearner(
        _ScriptTrainer(), MODEL, eval_model=before, eval_after=after, workdir=str(tmp_path)
    )
    events = []
    result = train(env, learner, rounds=1, group_size=1, batch=1, on_round=events.append)
    assert result.baseline == 0.0 and result.best == 1.0
    assert learner.checkpoint == str(tmp_path / "round1" / "checkpoint")
    assert events[0].kept


def test_trainers_are_learners_by_name(tmp_path):
    learner = build(
        "learner", {"use": "trl", "model": MODEL, "workdir": str(tmp_path), "max_steps": 5}
    )
    assert isinstance(learner, ExternalLearner) and learner.trainer.id == "trl"
    with pytest.raises(ValueError, match="eval_model"):
        learner.policy()


def test_run_streams_metrics_and_logs(tmp_path):
    seen = []
    trainer = _ScriptTrainer()
    result = trainer.run(
        trainer.prepare(None, MODEL, {}, tmp_path), lambda kind, payload: seen.append(kind)
    )
    assert isinstance(result, JobResult) and result.metrics == [{"step": 10, "reward": 0.75}]
    assert "metric" in seen
    assert trainer.artifacts(trainer.prepare(None, MODEL, {}, tmp_path))[0]["kind"] == "checkpoint"


def test_builtin_nenyax_trainer_runs_for_real(tmp_path):
    env = _quiz()
    answers = {t.prompt: t.metadata["answer"] for t in env.tasks()}
    trainer = get("nenyax")
    job = trainer.prepare(
        env,
        "solver",
        {
            "rounds": 1,
            "group_size": 2,
            "batch": 2,
            "model_spec": FunctionModel(lambda m: answers.get(m[-1].content, "0")),
        },
        tmp_path,
    )
    assert json.loads(job.files["plan"].read_text())["learner"]["use"] == "incontext"
    result = trainer.run(job)
    assert result.metrics[-1]["baseline"] == 1.0


def test_completion_text_shapes():
    assert completion_text("x") == "x"
    assert (
        completion_text(
            [{"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}]
        )
        == "b"
    )
    assert completion_text({"content": None}) == ""
