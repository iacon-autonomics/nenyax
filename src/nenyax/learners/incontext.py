"""In-context learning from experience: works with *any* chat model, open or closed.

No weights change. The learner keeps a bank of the best-scoring episodes and shows the most
relevant ones to the model as worked examples. For API models whose weights you can't touch,
this is how a policy "learns"; for open models it is a strong, cheap baseline before RL.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

from ..policy import ChatModel
from ..types import Message, Trajectory
from .base import has_signal


def _exchange(traj: Trajectory) -> list[Message] | None:
    """The episode's conversation without system prompts, ending in an assistant reply."""
    turns = [m for m in traj.messages if m.role != "system"]
    if not turns or turns[-1].role != "assistant":
        return None
    return turns


class ExampleBank:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.items: list[tuple[float, str, list[Message]]] = []  # (score, key, turns)

    def add(self, score: float, key: str, turns: list[Message]) -> None:
        self.items = [it for it in self.items if it[1] != key or it[0] > score]
        if not any(it[1] == key for it in self.items):
            self.items.append((score, key, turns))
        self.items.sort(key=lambda it: -it[0])
        del self.items[self.capacity :]


class _WithExamples:
    """A ChatModel that prepends banked examples as prior conversation turns."""

    def __init__(self, model: ChatModel, bank: ExampleBank, shots: int) -> None:
        self.model, self.bank, self.shots = model, bank, shots
        self.model_name = getattr(model, "model", None)

    def complete(self, messages: Sequence[Message], **params: Any) -> Message:
        query = next((m.content for m in reversed(messages) if m.role == "user"), "")
        examples = [turns for _, key, turns in self.bank.items if key != query][: self.shots]
        system = [m for m in messages if m.role == "system"]
        rest = [m for m in messages if m.role != "system"]
        shots = [m for turns in examples for m in turns]
        return self.model.complete([*system, *shots, *rest], **params)


class InContextLearner:
    """Learn by curating worked examples from the model's own successful episodes."""

    def __init__(
        self, model: ChatModel, *, shots: int = 3, capacity: int = 32, min_score: float = 1.0
    ) -> None:
        self.model, self.shots, self.min_score = model, shots, min_score
        self.bank = ExampleBank(capacity)

    def policy(self) -> ChatModel:
        return _WithExamples(self.model, self.bank, self.shots)

    def update(self, groups: list[list[Trajectory]]) -> dict[str, float]:
        added = 0
        for group in groups:
            best = max(group, key=lambda t: t.score or 0.0)
            turns = _exchange(best)
            if turns and (best.score or 0.0) >= self.min_score:
                key = next((m.content for m in turns if m.role == "user"), "") or best.env_id
                self.bank.add(best.score or 0.0, str(key), turns)
                added += 1
        return {
            "examples_added": float(added),
            "bank_size": float(len(self.bank.items)),
            "signal": has_signal(groups),
        }

    def snapshot(self) -> list:
        return copy.deepcopy(self.bank.items)

    def restore(self, state: list) -> None:
        self.bank.items = copy.deepcopy(state)
