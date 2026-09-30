import io
import json

import pytest
from conftest import CountdownEnv, EchoJudgeEnv

import nenyax
from nenyax import (
    Endpoint,
    FunctionModel,
    Judgment,
    ModelServer,
    NativeAgent,
    PolicyMismatch,
    Tier,
    Trajectory,
    UnsupportedCapability,
    check,
)
from nenyax.registry import DriverNotFound, parse_uri
from nenyax.testing import assert_conforms

# -- contract types ------------------------------------------------------------------------------


def test_judgment_rejects_non_finite_scores():
    with pytest.raises(ValueError):
        Judgment(score=float("nan"))


def test_trajectory_round_trips_through_json(countdown):
    traj = countdown.rollout(countdown.reference_policy(), seed=3)
    again = Trajectory.model_validate_json(traj.model_dump_json())
    assert again == traj and again.score == 1.0


# -- step mode -----------------------------------------------------------------------------------


def test_step_rollout_scores_reference_and_null(countdown):
    assert countdown.rollout(countdown.reference_policy(), seed=4).score == 1.0
    assert countdown.rollout(countdown.null_policy(), seed=4).score == 0.0


def test_chat_model_drives_text_step_env(countdown):
    model = FunctionModel(lambda msgs: msgs[-1].content.removeprefix("Say "))
    traj = countdown.rollout(model, seed=2)
    assert traj.score == 1.0
    assert [m.role for m in traj.messages] == ["user", "assistant"]


def test_chat_model_on_non_text_env_is_a_policy_mismatch():
    env = CountdownEnv(text=False)
    with pytest.raises(PolicyMismatch):
        env.rollout(FunctionModel(lambda m: "x"))


def test_rollout_mode_cannot_be_stepped(echo_env):
    with pytest.raises(UnsupportedCapability):
        echo_env.session()


def test_rollout_errors_are_captured_not_raised(countdown):
    def boom(obs):
        raise RuntimeError("policy crashed")

    traj = countdown.rollout(boom, seed=1)
    assert not traj.ok and "policy crashed" in traj.error


# -- model-driven mode and the recording server --------------------------------------------------


def test_model_driven_rollout_records_every_model_call(echo_env):
    task = echo_env.tasks()[1]
    traj = echo_env.rollout(echo_env.reference_policy(), task=task)
    assert traj.score == 1.0
    assert len(traj.model_calls) == 1
    call = traj.model_calls[0]
    assert call.request["messages"][0]["content"] == "Repeat exactly: river"
    assert call.response["choices"][0]["message"]["content"] == "river"


def test_native_agent_is_reported_unsupported_when_format_has_none(echo_env):
    traj = echo_env.rollout(NativeAgent("oracle"))
    assert "UnsupportedCapability" in traj.error


def test_server_proxies_to_upstream_and_records_both_sides():
    upstream = ModelServer(FunctionModel(lambda m: "pong"), model_name="up").start()
    try:
        proxy = ModelServer(Endpoint(upstream.base_url(), "up")).start()
        try:
            sid = proxy.new_session()
            reply = Endpoint(proxy.base_url(sid), "anything").complete(
                [nenyax.Message(role="user", content="ping")]
            )
            assert reply.content == "pong"
            assert len(proxy.calls(sid)) == 1 and proxy.calls("other") == []
            assert upstream.calls()[0].request["model"] == "up"
        finally:
            proxy.stop()
    finally:
        upstream.stop()


def test_server_streams_when_asked():
    import urllib.request

    with ModelServer(FunctionModel(lambda m: "hi")) as server:
        req = urllib.request.Request(
            server.base_url() + "/chat/completions",
            data=json.dumps(
                {"model": "x", "stream": True, "messages": [{"role": "user", "content": "yo"}]}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        body = urllib.request.urlopen(req, timeout=5).read().decode()
    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
    assert events[0]["choices"][0]["delta"]["content"] == "hi"
    assert body.rstrip().endswith("data: [DONE]")


def test_server_reports_backend_errors_as_500_and_records_them():
    def fail(messages):
        raise ValueError("model exploded")

    with ModelServer(FunctionModel(fail)) as server:
        with pytest.raises(RuntimeError, match="500"):
            Endpoint(server.base_url(), "x").complete([nenyax.Message(role="user", content="?")])
        assert "model exploded" in server.calls()[0].error


# -- conformance ---------------------------------------------------------------------------------


def test_good_step_env_is_audited(countdown):
    assert_conforms(countdown, "audited")


def test_good_rollout_env_is_audited(echo_env):
    assert_conforms(echo_env, "audited")


def test_lenient_verifier_fails_the_audit():
    report = check(CountdownEnv(lenient=True))
    assert report.tier == Tier.TRAINABLE
    failed = {c.name for c in report.checks if c.status == "fail"}
    assert failed == {"verifier_discriminates"}


def test_env_that_bypasses_the_endpoint_is_not_trainable():
    report = check(EchoJudgeEnv(bypass=True))
    assert report.tier == Tier.EVALUABLE
    signal = next(c for c in report.checks if c.name == "learning_signal")
    assert signal.status == "fail" and "bypassed" in signal.detail


# -- router and runner ---------------------------------------------------------------------------


def test_parse_uri_coerces_query_values():
    assert parse_uri("gymnasium:CartPole-v1?max_episode_steps=50&render_mode=null") == (
        "gymnasium",
        "CartPole-v1",
        {"max_episode_steps": 50, "render_mode": None},
    )


def test_unknown_uri_is_a_clear_error():
    with pytest.raises(DriverNotFound, match="gymnasium"):
        nenyax.load("definitely-not-a-driver-or-path")


def test_registered_driver_is_routable():
    class ToyDriver(nenyax.Driver):
        name, format = "toy", "Toy"

        def load(self, target, **options):
            return CountdownEnv(**options)

    nenyax.register(ToyDriver())
    env = nenyax.load("toy:countdown?lenient=true")
    assert env.lenient is True


def test_run_streams_jsonl_and_supports_policy_factories(echo_env):
    buf = io.StringIO()
    trajs = list(nenyax.run(echo_env, echo_env.reference_policy, concurrency=3, out=buf))
    assert [t.score for t in trajs] == [1.0, 1.0, 1.0]
    lines = buf.getvalue().strip().splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["env_id"] == "toy:echo-judge"


def test_chat_model_in_step_mode_records_calls_with_tokens(countdown, monkeypatch):
    def fake_chat(self, body):
        target = body["messages"][-1]["content"].removeprefix("Say ")
        return {
            "prompt_token_ids": [1, 2, 3],
            "choices": [
                {
                    "message": {"role": "assistant", "content": target},
                    "token_ids": [7, 8],
                    "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]},
                }
            ],
        }

    monkeypatch.setattr(Endpoint, "chat", fake_chat)
    traj = countdown.rollout(Endpoint("http://unused/v1", "m"), seed=5)
    assert traj.score == 1.0
    (call,) = traj.model_calls
    assert call.prompt_token_ids == [1, 2, 3] and call.completion_token_ids == [7, 8]
    assert call.completion_logprobs == [-0.1, -0.2]
