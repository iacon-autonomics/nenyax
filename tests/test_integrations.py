import os

import pytest

import nenyax
from nenyax import integrations as I
from nenyax.environment import NenyaxError
from nenyax.integrations import export as X

NETWORK = pytest.mark.skipif(os.environ.get("NENYAX_NETWORK") != "1", reason="set NENYAX_NETWORK=1")


def test_hosted_provider_reports_missing_key(monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    assert I.providers()["together"].missing() == ["set TOGETHER_API_KEY"]
    with pytest.raises(NenyaxError, match="TOGETHER_API_KEY"):
        I.endpoint("together", "Qwen/Qwen3-8B")


def test_provider_endpoint_uses_key_and_base_url(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    ep = I.endpoint("openrouter", "qwen/qwen3-8b", temperature=0.2)
    assert ep.base_url == "https://openrouter.ai/api/v1" and ep.api_key == "sk-test"
    assert ep.default_params == {"temperature": 0.2}
    assert I.endpoint("vllm", "m").api_key == "EMPTY"  # local servers need no key


def test_explicit_key_overrides_environment(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert I.endpoint("groq", "m", api_key="sk-given").api_key == "sk-given"


def _trajs():
    env = nenyax.load("reasoning_gym:leg_counting?size=4&seed=2") if _has_rg() else None
    if env is None:
        pytest.skip("reasoning_gym not installed")
    right = [
        env.rollout(
            nenyax.FunctionModel(lambda m, i=i: f"<answer>{env.dataset[i]['answer']}</answer>"),
            task=t,
        )
        for i, t in enumerate(env.tasks())
    ]
    wrong = [
        env.rollout(nenyax.FunctionModel(lambda m: "<answer>-1</answer>"), task=t)
        for t in env.tasks()
    ]
    return right, wrong


def _has_rg():
    from importlib.util import find_spec

    return find_spec("reasoning_gym") is not None


def test_export_shapes():
    right, wrong = _trajs()
    assert len(X.to_rows(right + wrong, "raw")) == 8
    sft = X.to_rows(right + wrong, "sft")
    assert len(sft) == 4 and all(r["score"] == 1.0 for r in sft)
    dpo = X.to_rows(right + wrong, "dpo")
    assert len(dpo) == 4
    assert dpo[0]["chosen"][-1]["role"] == "assistant" and dpo[0]["prompt"][-1]["role"] == "user"


def test_callbacks_see_every_episode():
    right, _ = _trajs()
    env = nenyax.load("reasoning_gym:leg_counting?size=4&seed=2")
    seen = []

    class Collect:
        def on_trajectory(self, traj):
            seen.append(traj.score)

    list(nenyax.run(env, env.reference_policy(), n=4, callbacks=[Collect()]))
    assert seen == [1.0, 1.0, 1.0, 1.0]


@NETWORK
@pytest.mark.parametrize(
    "hub,query", [("huggingface", "wordle"), ("prime", "math"), ("harbor", "terminal-bench")]
)
def test_live_hub_search_returns_loadable_uris(hub, query):
    hits = I.search(hub, query, limit=3)
    assert hits and all(h.uri.split(":", 1)[0] in nenyax.drivers() for h in hits)
