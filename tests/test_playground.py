"""The playground: describe, list tasks, reset, step, sample and judge any step environment."""

from __future__ import annotations

import nenyax
from nenyax.playground import Playground
from nenyax.registry import load


def _quiz():
    return nenyax.define(
        "quiz",
        tasks=[{"prompt": "2+2?", "answer": "4"}, {"prompt": "3+3?", "answer": "6"}],
        score=lambda t, r: float(r.strip() == t["answer"]),
    )


def test_single_turn_episode_by_hand():
    pg = Playground(_quiz())
    d = pg.handle("describe")
    assert d["steppable"] and d["num_tasks"] == 2 and d["tools"] == []
    tasks = pg.handle("tasks")
    assert [t["prompt"] for t in tasks["items"]] == ["2+2?", "3+3?"]
    first = pg.handle("reset", {"task": "1"})
    assert first["observation"]["text"] == "3+3?" and not first["done"]
    out = pg.handle("step", {"action": "6"})
    assert out["reward"] == 1.0 and out["done"]
    assert pg.handle("judge")["score"] == 1.0
    assert "error" in pg.handle("step", {"action": "6"})  # the episode is over


def test_errors_come_back_instead_of_raising():
    pg = Playground(_quiz())
    assert "reset first" in pg.handle("step", {"action": "x"})["error"]
    assert "no task" in pg.handle("reset", {"task": "99"})["error"]
    assert "unknown operation" in pg.handle("fly")["error"]


def test_gymnasium_has_legal_actions_samples_and_a_rendered_board():
    pg = Playground(load("gymnasium:FrozenLake-v1"))
    state = pg.handle("reset", {"seed": 0})
    assert state["legal_actions"] == [0, 1, 2, 3]
    assert "S" in state["render"]["text"]  # the lake, drawn as text
    assert pg.handle("sample")["action"] in (0, 1, 2, 3)
    assert pg.handle("step", {"action": 1})["steps"] == 1
