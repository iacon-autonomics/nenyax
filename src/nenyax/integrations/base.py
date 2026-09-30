"""What every integration shares: a name, a kind, the credentials it needs, and honest status.

Credentials are never stored by Nenyax. They come from environment variables or are passed
explicitly, and ``missing()`` says exactly what is absent.
"""

from __future__ import annotations

import os
from typing import Literal

Kind = Literal["provider", "hub", "telemetry", "exporter", "trainer", "sandbox"]


class Integration:
    name: str
    kind: Kind
    #: Environment variables required (bring-your-own-key services).
    credentials: tuple[str, ...] = ()
    #: Python modules required (vendor SDKs, imported lazily).
    modules: tuple[str, ...] = ()
    #: One-line description shown by ``nenyax integrations``.
    summary: str = ""
    #: Whether this integration has been exercised against the live service.
    verified_live: bool = False

    def __init__(self, **credentials: str) -> None:
        self._given = {k.upper(): v for k, v in credentials.items()}

    def credential(self, var: str) -> str | None:
        return self._given.get(var) or os.environ.get(var)

    def missing(self) -> list[str]:
        from importlib.util import find_spec

        out = [f"pip install {m}" for m in self.modules if find_spec(m) is None]
        return out + [f"set {v}" for v in self.credentials if not self.credential(v)]

    def ready(self) -> bool:
        return not self.missing()
