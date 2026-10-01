"""``nenyax:`` driver: environments pushed to a Nenyax platform with ``nenyax push``.

    nenyax try nenyax:dev/gsm8k-mini            # latest version
    nenyax try nenyax:dev/gsm8k-mini@0.1.0      # a pinned version

Uses the credentials from ``nenyax auth login`` (public environments load without them).
"""

from __future__ import annotations

import json
from typing import Any

from ..environment import Environment
from ..registry import Driver


class NenyaxHubDriver(Driver):
    name = "nenyax"
    module = None
    format = "Nenyax platform package (nenyax push)"

    def load(self, target: str, **options: Any) -> Environment:
        from .. import platform

        slug, _, version = target.partition("@")
        try:
            url, token = platform.credentials()
        except platform.PlatformError:
            url, token = "http://127.0.0.1:4300/api", ""
        listing = json.loads(platform._request(f"{url}/listings/{slug}", token))
        fetch = dict((listing.get("stats") or {}).get("fetch") or {})
        if fetch.get("hub") != "nenyax":
            raise platform.PlatformError(f"{slug} is not a pushed Nenyax environment")
        if version:
            fetch["version"] = version
        return platform.load_package(fetch, url, token)
