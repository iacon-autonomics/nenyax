from __future__ import annotations

import asyncio
import math
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def to_json(x: Any, _depth: int = 0) -> Any:
    """Best-effort conversion of native values (numpy, tuples, dataclasses) to JSON values."""
    if _depth > 32:
        return repr(x)
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else repr(x)
    if hasattr(x, "tolist"):  # numpy arrays and scalars
        return to_json(x.tolist(), _depth + 1)
    if isinstance(x, dict):
        return {str(k): to_json(v, _depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [to_json(v, _depth + 1) for v in x]
    if hasattr(x, "model_dump"):
        return to_json(x.model_dump(), _depth + 1)
    if hasattr(x, "__dataclass_fields__"):
        return {f: to_json(getattr(x, f), _depth + 1) for f in x.__dataclass_fields__}
    return repr(x)


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine to completion from sync code, even if an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    result: dict[str, Any] = {}

    def target() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as e:
            result["error"] = e

    t = threading.Thread(target=target)
    t.start()
    t.join()
    if "error" in result:
        raise result["error"]
    return result["value"]
