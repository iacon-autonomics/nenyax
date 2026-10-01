"""Playing rollout-mode environments as the model: turns, text replies, tool calls, scoring."""

from __future__ import annotations

import json
import threading

import pytest

from nenyax.environment import ModelDrivenEnvironment
from nenyax.human import HumanAborted, HumanModel
from nenyax.playground import Playground
from nenyax.types import Capabilities, Judgment, Manifest, Message, Mode, TaskRef

CALC = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}


class ToolQuiz(ModelDrivenEnvironment):
    """A rollout-mode environment that talks to the model over HTTP, like Verifiers or Harbor:
    it offers a calculator tool, answers the call, then scores the final reply."""

    def __init__(self) -> None:
        self.manifest = Manifest(
            id="test:toolquiz",
            name="toolquiz",
            source_format="test",
            mode=Mode.ROLLOUT,
            capabilities=Capabilities(text=True),
            num_tasks=1,
        )

    def tasks(self, split=None, limit=None):
        return [
            TaskRef(id="0", prompt="What is 6 * 7? Use the calculator.", metadata={"answer": "42"})
        ]

    def _run(self, endpoint, *, task, seed, max_steps):
        messages = [
            {"role": "system", "content": "You can call tools."},
            {"role": "user", "content": "What is 6 * 7? Use the calculator."},
        ]
        first = endpoint.chat({"messages": messages, "tools": [CALC], "temperature": 0.0})
        msg = first["choices"][0]["message"]
        messages.append(msg)
        for call in msg.get("tool_calls") or []:
            expr = json.loads(call["function"]["arguments"])["expression"]
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": str(eval(expr))}
            )  # noqa: S307
        second = endpoint.chat({"messages": messages, "tools": [CALC]})
        final = second["choices"][0]["message"]["content"] or ""
        messages.append(second["choices"][0]["message"])
        score = float(final.strip().endswith("42"))
        used_tool = float(bool(msg.get("tool_calls")))
        judgment = Judgment(score=score, components={"correct": score, "used_tool": used_tool})
        return judgment, [Message.model_validate(m) for m in messages], {}


def test_human_model_blocks_until_answered():
    human = HumanModel(timeout_s=5)
    out: dict = {}
    t = threading.Thread(
        target=lambda: out.update(
            m=human.complete([Message(role="user", content="hi")], tools=[CALC])
        )
    )
    t.start()
    turn = human.wait_for_turn(2)
    assert (
        turn is not None
        and turn.tools == [CALC]
        and turn.public()["messages"][0]["content"] == "hi"
    )
    human.answer(turn.id, content="hello")
    t.join(2)
    assert out["m"].content == "hello"
    with pytest.raises(ValueError):
        human.answer(content="again")  # nothing is waiting


def test_abort_unblocks_the_environment():
    human = HumanModel(timeout_s=5)
    errors: list = []

    def ask():
        try:
            human.complete([Message(role="user", content="hi")])
        except HumanAborted as e:
            errors.append(e)

    t = threading.Thread(target=ask)
    t.start()
    assert human.wait_for_turn(2) is not None
    human.abort()
    t.join(2)
    assert errors


def test_play_a_tool_using_rollout_environment_as_the_model():
    pg = Playground(ToolQuiz())
    d = pg.handle("describe")
    assert d["playable_as_model"] and not d["steppable"]

    first = pg.handle("start_rollout", {"task": "0"})
    assert first["status"] == "your_turn"
    turn = first["turn"]
    assert turn["tools"][0]["function"]["name"] == "calculator"
    assert [m["role"] for m in turn["messages"]] == ["system", "user"]

    call = {"name": "calculator", "arguments": {"expression": "6*7"}}
    second = pg.handle("reply", {"tool_calls": [call]})
    assert second["status"] == "your_turn", second
    assert second["turn"]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": second["turn"]["messages"][-2]["tool_calls"][0]["id"],
        "content": "42",
    }

    done = pg.handle("reply", {"content": "The answer is 42"})
    assert done["status"] == "done" and done["score"] == 1.0
    assert done["judgment"]["components"] == {"correct": 1.0, "used_tool": 1.0}
    assert done["model_calls"] == 2


def test_abort_mid_episode_and_restart():
    pg = Playground(ToolQuiz())
    assert pg.handle("start_rollout")["status"] == "your_turn"
    assert pg.handle("abort") == {"aborted": True}
    again = pg.handle("start_rollout")
    assert again["status"] == "your_turn" and again["turn"]["index"] == 0
    assert "error" in pg.handle("reply", {})  # empty replies are refused
    pg.handle("close")


def test_step_environments_say_to_use_step():
    import nenyax

    pg = Playground(nenyax.define("q", tasks=[{"prompt": "x"}], score=lambda t, r: 1.0))
    assert not pg.handle("describe")["playable_as_model"]
    assert "reset and step" in pg.handle("start_rollout")["error"]
