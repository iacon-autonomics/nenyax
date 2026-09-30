"""Every built-in algorithm: advantage math exactly, and a real (tiny-model) update end to end."""

from importlib.util import find_spec

import pytest

import nenyax
from nenyax import config
from nenyax.learners import advantages as A

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
needs_torch = pytest.mark.skipif(
    find_spec("torch") is None or find_spec("transformers") is None, reason="needs nenyax[train]"
)


def test_advantage_estimators():
    groups = [[1.0, 0.0, 0.0, 1.0], [1.0, 1.0]]
    assert A.grpo(groups) == [[1.0, -1.0, -1.0, 1.0], [0.0, 0.0]]
    assert A.dr_grpo(groups) == [[0.5, -0.5, -0.5, 0.5], [0.0, 0.0]]
    # leave-one-out: 1 - mean(0, 0, 1) = 2/3 ; 0 - mean(1, 0, 1) = -2/3
    assert [round(a, 4) for a in A.rloo(groups)[0]] == [0.6667, -0.6667, -0.6667, 0.6667]
    # batch mean over all six scores is 4/6; the all-equal group still teaches nothing
    assert [round(a, 4) for a in A.reinforce(groups)[0]] == [0.3333, -0.6667, -0.6667, 0.3333]
    assert A.reinforce(groups)[1] == [0.0, 0.0]
    assert A.batch_norm(groups)[0] == [1.0, -1.0, -1.0, 1.0]


def _env():
    tasks = [{"prompt": f"Say {i}.", "answer": str(i)} for i in range(4)]
    return nenyax.define("tiny", tasks=tasks, score=lambda t, r: float(t["answer"] in r))


def _groups(learner):
    trajs = list(nenyax.run(_env(), learner.policy, n=4))
    assert all(t.ok for t in trajs), trajs[0].error
    for t, s in zip(trajs, [1.0, 0.0, 1.0, 0.0], strict=True):
        t.judgment.score = s  # force a known signal so every algorithm has something to learn
    return [trajs[:2], trajs[2:]]


@needs_torch
@pytest.mark.parametrize(
    "name", ["grpo", "dr_grpo", "dapo", "rloo", "reinforce", "reinforce_pp", "gspo", "rft", "dpo"]
)
def test_every_algorithm_updates_weights(name):
    import torch

    spec = {"use": name, "model": TINY, "lr": 1e-3, "max_new_tokens": 6, "device": "cpu"}
    learner = config.build("learner", spec)
    before = {k: v.clone() for k, v in learner.model.state_dict().items()}
    metrics = learner.update(_groups(learner))
    assert all(isinstance(v, float) for v in metrics.values()), metrics
    changed = any(not torch.equal(before[k], v) for k, v in learner.model.state_dict().items())
    assert changed, f"{name} did not change the weights: {metrics}"


@needs_torch
def test_ppo_epochs_and_kl_run():
    learner = config.build(
        "learner",
        {
            "use": "pg",
            "model": TINY,
            "loss": "ppo",
            "epochs": 2,
            "kl": 0.05,
            "lr": 1e-3,
            "max_new_tokens": 6,
            "device": "cpu",
            "clip": [0.2, 0.28],
        },
    )
    metrics = learner.update(_groups(learner))
    assert metrics["samples"] == 4 and 0.0 <= metrics["clip_frac"] <= 1.0


@needs_torch
def test_learner_conformance_for_torch_learners():
    from nenyax.testing import assert_learner_conforms

    for name in ("dapo", "rft", "dpo"):
        assert_learner_conforms(
            config.build(
                "learner", {"use": name, "model": TINY, "max_new_tokens": 6, "device": "cpu"}
            )
        )
