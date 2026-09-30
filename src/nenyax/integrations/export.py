"""Export trajectories as training data: raw records, SFT conversations, or DPO pairs.

    rows = to_rows(trajectories, format="sft", min_score=1.0)
    to_parquet(rows, "sft.parquet")
    push_to_hub(rows, "me/my-rl-data")          # needs HF_TOKEN

``format``:
    raw  one row per episode, every field (JSON-encoded where nested)
    sft  successful conversations only: {"messages": [...]} with score >= ``min_score``
    dpo  per task, best vs. worst episode: {"prompt", "chosen", "rejected"} (needs >= 2 scores)
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from ..environment import NenyaxError
from ..types import Trajectory


def _messages(traj: Trajectory) -> list[dict[str, Any]]:
    return [m.to_openai() for m in traj.messages]


def to_rows(
    trajectories: Iterable[Trajectory],
    format: Literal["raw", "sft", "dpo"] = "raw",
    *,
    min_score: float = 1.0,
) -> list[dict[str, Any]]:
    trajs = [t for t in trajectories if t.ok]
    if format == "raw":
        return [
            {
                "env_id": t.env_id,
                "mode": t.mode.value,
                "task_id": t.task.id if t.task else None,
                "seed": t.seed,
                "score": t.score,
                "messages": json.dumps(_messages(t)),
                "steps": json.dumps([s.model_dump() for s in t.steps]),
                "judgment": t.judgment.model_dump_json() if t.judgment else None,
                "model_calls": len(t.model_calls),
                "duration_s": t.duration_s,
            }
            for t in trajs
        ]
    if format == "sft":
        return [
            {"messages": _messages(t), "env_id": t.env_id, "score": t.score}
            for t in trajs
            if (t.score or 0.0) >= min_score and t.messages
        ]
    if format == "dpo":
        groups: dict[tuple, list[Trajectory]] = defaultdict(list)
        for t in trajs:
            if t.messages:
                groups[(t.env_id, t.task.id if t.task else t.seed)].append(t)
        rows = []
        for group in groups.values():
            best = max(group, key=lambda t: t.score)
            worst = min(group, key=lambda t: t.score)
            if best.score <= worst.score:
                continue
            split = _prompt_len(best)
            rows.append(
                {
                    "prompt": _messages(best)[:split],
                    "chosen": _messages(best)[split:],
                    "rejected": _messages(worst)[_prompt_len(worst) :],
                    "score_chosen": best.score,
                    "score_rejected": worst.score,
                }
            )
        return rows
    raise ValueError(f"unknown format {format!r}")


def _prompt_len(traj: Trajectory) -> int:
    """Messages before the first assistant turn form the prompt."""
    for i, m in enumerate(traj.messages):
        if m.role == "assistant":
            return i
    return len(traj.messages)


def to_jsonl(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def to_dataset(rows: list[dict[str, Any]]) -> Any:
    try:
        from datasets import Dataset
    except ImportError as e:
        raise NenyaxError("pip install datasets") from e
    return Dataset.from_list(rows)


def to_parquet(rows: list[dict[str, Any]], path: str | Path) -> Path:
    to_dataset(rows).to_parquet(str(path))
    return Path(path)


def push_to_hub(
    rows: list[dict[str, Any]], repo_id: str, *, private: bool = True, token: str | None = None
) -> str:
    """Upload to a Hugging Face dataset repo (private by default). Needs ``HF_TOKEN``."""
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise NenyaxError("push_to_hub needs HF_TOKEN (or token=...)")
    to_dataset(rows).push_to_hub(repo_id, private=private, token=token)
    return f"https://huggingface.co/datasets/{repo_id}"
