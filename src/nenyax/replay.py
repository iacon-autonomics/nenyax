"""Human-readable replay of one episode: what the agent saw, what it did, how it was judged."""

from __future__ import annotations

import json

from .types import Trajectory


def _clip(text: object, width: int) -> str:
    s = text if isinstance(text, str) else json.dumps(text, default=str)
    s = s.strip()
    return s if len(s) <= width else s[: width - 1] + "…"


def render(traj: Trajectory, width: int = 400) -> str:
    lines = [
        f"━━ {traj.env_id}  [{traj.mode.value}]  task={traj.task.id if traj.task else '-'}"
        f"  seed={traj.seed}  {traj.duration_s or 0:.2f}s"
    ]
    if traj.messages:
        for m in traj.messages:
            content = (
                m.content if m.content is not None else [t.model_dump() for t in m.tool_calls or []]
            )
            lines.append(f"  {m.role:>9} │ {_clip(content, width)}")
    elif traj.steps:
        if traj.initial_observation:
            obs = traj.initial_observation
            lines.append(f"  {'observe':>9} │ {_clip(obs.text or obs.data, width)}")
        for i, s in enumerate(traj.steps):
            lines.append(f"  {'act ' + str(i):>9} │ {_clip(s.action, width)}   reward={s.reward:g}")
            if not s.done:
                lines.append(
                    f"  {'observe':>9} │ {_clip(s.observation.text or s.observation.data, width)}"
                )
    if traj.model_calls:
        tokens = sum(1 for c in traj.model_calls if c.completion_token_ids)
        lines.append(
            f"  {'model':>9} │ {len(traj.model_calls)} calls recorded"
            f"{f', {tokens} with token ids' if tokens else ''}"
        )
    if traj.error:
        lines.append(f"  {'ERROR':>9} │ {_clip(traj.error, width)}")
    if traj.judgment:
        j = traj.judgment
        parts = f"  {j.components}" if j.components else ""
        lines.append(f"  {'judged':>9} │ score={j.score:g}  by {j.judge or j.source}{parts}")
    return "\n".join(lines)
