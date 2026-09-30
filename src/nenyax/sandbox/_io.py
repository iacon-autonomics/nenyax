"""Bounded output capture for sandboxed commands.

Two hazards: an agent can print gigabytes, and a command can leave background processes holding
its output pipes open (``server &``). We keep the first ``cap`` bytes of each stream, keep
draining the rest so the child never blocks, and stop waiting for the pipes shortly after the
command itself exits.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import IO

DEFAULT_OUTPUT_CAP = 1 << 20  # 1 MiB per stream
PIPE_GRACE_S = 2.0


@dataclass
class Captured:
    stdout: bytes
    stderr: bytes
    returncode: int
    truncated: bool
    timed_out: bool


def run_bounded(
    proc: subprocess.Popen,
    *,
    timeout: float | None,
    kill: Callable[[], None],
    cap: int = DEFAULT_OUTPUT_CAP,
    stdin: bytes | None = None,
) -> Captured:
    """Wait for ``proc`` with bounded output; call ``kill()`` if it exceeds ``timeout``."""
    buffers = {"out": bytearray(), "err": bytearray()}
    truncated = threading.Event()
    lock = threading.Lock()

    def drain(stream: IO[bytes], key: str) -> None:
        try:
            while chunk := stream.read1(65536):  # whatever is available, don't wait to fill
                with lock:
                    room = cap - len(buffers[key])
                    if room > 0:
                        buffers[key] += chunk[:room]
                if len(chunk) > max(room, 0):
                    truncated.set()
        except (OSError, ValueError):
            pass

    readers = [
        threading.Thread(target=drain, args=(proc.stdout, "out"), daemon=True),
        threading.Thread(target=drain, args=(proc.stderr, "err"), daemon=True),
    ]
    for t in readers:
        t.start()
    if proc.stdin is not None:
        try:
            if stdin:
                proc.stdin.write(stdin)
            proc.stdin.close()
        except BrokenPipeError:
            pass
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill()
        proc.wait()
    # Background children may hold the pipes open; give both streams one shared grace period.
    deadline = time.monotonic() + PIPE_GRACE_S
    for t in readers:
        t.join(max(0.0, deadline - time.monotonic()))
    with lock:
        return Captured(
            bytes(buffers["out"]),
            bytes(buffers["err"]),
            proc.returncode,
            truncated.is_set(),
            timed_out,
        )
