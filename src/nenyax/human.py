"""A person as the model: the ChatModel behind playing rollout- and task-mode environments by hand.

Rollout/task environments (Verifiers, Harbor, NeMo Gym, ...) call a model endpoint themselves.
:class:`HumanModel` is a :class:`~nenyax.policy.ChatModel` whose ``complete`` blocks until a
person answers: each request becomes a :class:`Turn` (the messages, the tools offered, the
sampling params), and :meth:`HumanModel.answer` replies with text or tool calls. Served through
the usual recording :class:`~nenyax.server.ModelServer`, the environment can't tell the
difference, so any of them can be played from the playground or a terminal.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .types import JSON, Message, ToolCall

#: Request params worth showing a person (the rest are transport details).
SHOWN_PARAMS = ("temperature", "max_tokens", "max_completion_tokens", "tool_choice", "stop")


class HumanAborted(RuntimeError):
    """The person stopped the episode (or never answered in time)."""


@dataclass
class Turn:
    id: str
    index: int
    messages: list[Message]
    tools: list[dict[str, JSON]]
    params: dict[str, JSON]
    asked_at: float = field(default_factory=time.time)
    reply: Message | None = None

    def public(self) -> dict[str, JSON]:
        return {
            "id": self.id,
            "index": self.index,
            "messages": [m.to_openai() for m in self.messages],
            "tools": self.tools,
            "params": self.params,
        }


class HumanModel:
    """A ChatModel answered by a person, one request at a time."""

    name = "you"

    def __init__(self, timeout_s: float = 600.0) -> None:
        self.timeout_s = timeout_s
        self.turns: list[Turn] = []
        self._cond = threading.Condition()
        self._aborted = False

    # -- the environment's side -------------------------------------------------------------

    def complete(self, messages: Sequence[Message], **params: Any) -> Message:
        with self._cond:
            if self._aborted:
                raise HumanAborted("the episode was stopped")
            turn = Turn(
                id=uuid.uuid4().hex[:12],
                index=len(self.turns),
                messages=list(messages),
                tools=list(params.get("tools") or []),
                params={k: params[k] for k in SHOWN_PARAMS if k in params},
            )
            self.turns.append(turn)
            self._cond.notify_all()
            deadline = time.monotonic() + self.timeout_s
            while turn.reply is None and not self._aborted:
                left = deadline - time.monotonic()
                if left <= 0:
                    self._aborted = True
                    raise HumanAborted("no reply in time")
                self._cond.wait(left)
            if turn.reply is None:
                raise HumanAborted("the episode was stopped")
            return turn.reply

    # -- the person's side ------------------------------------------------------------------

    @property
    def pending(self) -> Turn | None:
        with self._cond:
            last = self.turns[-1] if self.turns else None
            return last if last is not None and last.reply is None else None

    def wait_for_turn(self, timeout_s: float, after: int = 0) -> Turn | None:
        """Block until a turn at index >= ``after`` is waiting for an answer, or time runs out."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while True:
                last = self.turns[-1] if self.turns else None
                if last is not None and last.reply is None and last.index >= after:
                    return last
                left = deadline - time.monotonic()
                if left <= 0 or self._aborted:
                    return None
                self._cond.wait(min(left, 0.25))

    def answer(
        self,
        turn_id: str | None = None,
        *,
        content: str | None = None,
        tool_calls: list[dict[str, JSON]] | None = None,
    ) -> Turn:
        """Reply to the waiting turn with text, tool calls, or both."""
        if content is None and not tool_calls:
            raise ValueError("reply with content, tool_calls, or both")
        calls = [
            ToolCall.model_validate({"id": c.get("id") or f"call_{uuid.uuid4().hex[:8]}", **c})
            for c in tool_calls or []
        ]
        with self._cond:
            turn = self.turns[-1] if self.turns else None
            if turn is None or turn.reply is not None:
                raise ValueError("the environment is not waiting for a reply")
            if turn_id is not None and turn.id != turn_id:
                raise ValueError("that turn was already answered")
            turn.reply = Message(role="assistant", content=content, tool_calls=calls or None)
            self._cond.notify_all()
            return turn

    def abort(self) -> None:
        with self._cond:
            self._aborted = True
            self._cond.notify_all()
