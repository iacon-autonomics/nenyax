from importlib.util import find_spec

import pytest

import nenyax
from nenyax import FunctionModel, TrainingError, judges
from nenyax.learners import InContextLearner, group_advantages
from nenyax.replay import render
from nenyax.types import Judgment, Mode, Trajectory

TASKS = [{"prompt": f"Say the number {i} in answer tags.", "answer": str(i)} for i in range(12)]


def tagged(task, reply):
    return float(reply.strip() == f"<answer>{task['answer']}</answer>")


def env():
    return nenyax.define(
        "numbers", tasks=TASKS, score=tagged, reference=lambda t: f"<answer>{t['answer']}</answer>"
    )


class FormatLearnerModel:
    """Answers correctly only once it has seen a worked example (like a model learning format)."""

    def complete(self, messages, **params):
        number = messages[-1].content.split()[3]
        seen_example = any(m.role == "assistant" for m in messages[:-1])
        text = f"<answer>{number}</answer>" if seen_example else number
        return nenyax.Message(role="assistant", content=text)


class Lucky:
    """Right on the first try for task 0 only, so the first round has something to bank."""

    def __init__(self):
        self.inner = FormatLearnerModel()

    def complete(self, messages, **params):
        if messages[-1].content.startswith("Say the number 0 "):
            return nenyax.Message(role="assistant", content="<answer>0</answer>")
        return self.inner.complete(messages, **params)


# -- define and judges ---------------------------------------------------------------------------


def test_defined_env_is_audited():
    assert nenyax.check(env()).tier == nenyax.Tier.AUDITED


def test_multi_turn_defined_env():
    def step(state, action):
        state["n"] += 1
        return float(action == "stop"), action == "stop" or state["n"] >= 5

    e = nenyax.define(
        "stopper", reset=lambda seed: {"n": 0}, observe=lambda s: f"turn {s['n']}", step=step
    )
    actions = iter(["go", "go", "stop"])
    traj = e.rollout(lambda obs: next(actions))
    assert traj.score == 1.0 and len(traj.steps) == 3


def test_define_rejects_ambiguous_spec():
    with pytest.raises(ValueError):
        nenyax.define("bad", tasks=TASKS)


def test_weighted_judge_keeps_components():
    e = nenyax.with_judge(
        env(), judges.weighted({judges.EnvJudge(): 1.0, judges.Regex(r"^<answer>"): 0.5})
    )
    traj = e.rollout(FunctionModel(lambda m: "<answer>999</answer>"), task=e.tasks()[3])
    assert traj.score == 0.5
    assert traj.judgment.components == {"env": 0.0, "regex:^<answer>": 1.0}
    assert traj.judgment.details["env_judgment"]["score"] == 0.0


def test_exact_match_and_llm_judge():
    e = nenyax.with_judge(env(), judges.ExactMatch())
    assert e.rollout(FunctionModel(lambda m: "<answer> 4 </answer>"), task=e.tasks()[4]).score == 1

    grader = FunctionModel(lambda m: '{"score": 0.8, "reason": "mostly right"}')
    llm = nenyax.with_judge(env(), judges.LLMJudge(grader, rubric="Be correct."))
    traj = llm.rollout(FunctionModel(lambda m: "4"), task=llm.tasks()[4])
    assert traj.score == 0.8 and traj.judgment.details["reason"] == "mostly right"


# -- learning ------------------------------------------------------------------------------------


def test_group_advantages():
    def t(score):
        return Trajectory(env_id="x", mode=Mode.STEP, judgment=Judgment(score=score))

    advs = group_advantages([[t(1), t(0)], [t(1), t(1)]])
    assert advs[0] == [1.0, -1.0] and advs[1] == [0.0, 0.0]


def test_in_context_learner_improves_held_out_score():
    learner = InContextLearner(Lucky(), shots=1)
    result = nenyax.train(env(), learner, rounds=2, group_size=2, batch=12, seed=3)
    assert result.baseline == 0.0 and result.best == 1.0
    assert result.history[0].kept


class Saboteur:
    """A learner whose update makes things worse; the gate must revert it."""

    def __init__(self):
        self.good = True

    def policy(self):
        good = self.good
        return FunctionModel(
            lambda m: f"<answer>{m[-1].content.split()[3]}</answer>" if good else "nope"
        )

    def update(self, groups):
        self.good = False
        return {}

    def snapshot(self):
        return self.good

    def restore(self, state):
        self.good = state


def test_gate_reverts_updates_that_hurt():
    learner = Saboteur()
    result = nenyax.train(env(), learner, rounds=2, group_size=1, batch=2)
    assert [r.kept for r in result.history] == [False, False] and learner.good


def test_train_raises_when_every_episode_fails():
    broken = InContextLearner(FunctionModel(lambda m: (_ for _ in ()).throw(OSError("down"))))
    with pytest.raises(TrainingError, match="down"):
        nenyax.train(env(), broken, rounds=1)


def test_replay_renders_conversation_and_judgment():
    traj = env().rollout(FunctionModel(lambda m: "<answer>2</answer>"), task=env().tasks()[2])
    text = render(traj)
    assert "assistant │ <answer>2</answer>" in text and "score=1" in text


@pytest.mark.skipif(
    find_spec("torch") is None or find_spec("transformers") is None, reason="needs nenyax[train]"
)
def test_grpo_learner_updates_weights_from_trajectories():
    import torch

    from nenyax.learners import GRPOLearner

    learner = GRPOLearner(
        "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", lr=1e-3, max_new_tokens=6, device="cpu"
    )
    before = {k: v.clone() for k, v in learner.model.state_dict().items()}
    trajs = list(nenyax.run(env(), learner.policy, n=4))
    assert all(len(learner._samples(t)) == 1 for t in trajs)  # sampled ids ride along
    for t, s in zip(trajs, [1.0, 0.0, 1.0, 0.0], strict=True):
        t.judgment.score = s
    metrics = learner.update([trajs[:2], trajs[2:]])
    assert metrics["samples"] == 4
    assert any(not torch.equal(before[k], v) for k, v in learner.model.state_dict().items())


@pytest.mark.skipif(
    find_spec("tinker") is None or find_spec("transformers") is None,
    reason="needs nenyax[tinker,train]",
)
def test_tinker_learner_builds_valid_data_against_the_real_sdk_types():
    """Everything except the network: real tinker types, a fake service."""
    from types import SimpleNamespace

    import tinker
    from transformers import AutoTokenizer

    from nenyax.learners import TinkerLearner

    tok = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
    sent = []

    class Future:
        def __init__(self, value=None):
            self.value = value

        def result(self):
            return self.value

    class Sampler:
        def sample(self, prompt, num_samples, sampling_params):
            assert isinstance(prompt, tinker.ModelInput) and num_samples == 1
            seq = SimpleNamespace(tokens=tok.encode("<answer>3</answer>"), logprobs=[-0.5] * 5)
            return Future(SimpleNamespace(sequences=[seq]))

    class Trainer:
        def get_tokenizer(self):
            return tok

        def save_weights_and_get_sampling_client(self, name):
            return Sampler()

        def forward_backward(self, data, loss_fn):
            sent.append((data, loss_fn))
            return Future()

        def optim_step(self, adam_params):
            return Future()

    service = SimpleNamespace(create_lora_training_client=lambda base_model, rank: Trainer())
    learner = TinkerLearner("Qwen/Qwen3-8B", service_client=service)
    trajs = list(nenyax.run(env(), learner.policy, n=4))
    for t, s in zip(trajs, [1.0, 0.0, 1.0, 0.0], strict=True):
        t.judgment.score = s
    metrics = learner.update([trajs[:2], trajs[2:]])
    ((data, loss_fn),) = sent
    assert metrics["samples"] == 4 and loss_fn == "importance_sampling"
    datum = data[0]
    assert isinstance(datum, tinker.Datum)
    targets = datum.loss_fn_inputs["target_tokens"].to_numpy()
    advs = datum.loss_fn_inputs["advantages"].to_numpy()
    length = datum.model_input.length
    length = length() if callable(length) else length
    assert len(targets) == len(advs) == length
    assert advs[-1] == 1.0 and advs[0] == 0.0  # completion carries the advantage, prompt masked
