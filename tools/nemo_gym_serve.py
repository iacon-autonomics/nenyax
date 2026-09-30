"""Serve one NeMo Gym resources server standalone: no Ray, no head server, no per-server venv.

Run with a Python that has ``nemo-gym`` installed (it needs Python >= 3.13.14)::

    python tools/nemo_gym_serve.py reasoning_gym --port 8000

Then point Nenyax at it::

    nenyax check "nemo_gym:http://127.0.0.1:8000?data=<site-packages>/resources_servers/reasoning_gym/data/example.jsonl"

Only for servers that need no other servers (pure verifiers such as ``mcqa`` or ``reasoning_gym``).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import site
import sys
from pathlib import Path
from unittest.mock import MagicMock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("server", help="resources server name, e.g. mcqa or reasoning_gym")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn
    from nemo_gym.base_resources_server import BaseResourcesServerConfig, SimpleResourcesServer
    from nemo_gym.server_utils import ServerClient

    root = next(
        Path(p) / "resources_servers" / args.server
        for p in site.getsitepackages()
        if (Path(p) / "resources_servers" / args.server).is_dir()
    )
    sys.path.insert(0, str(root))
    app_module = importlib.import_module("app")

    def defined_here(base: type) -> type:
        found = [
            c
            for _, c in inspect.getmembers(app_module, inspect.isclass)
            if issubclass(c, base) and c is not base and c.__module__ == "app"
        ]
        if len(found) != 1:
            raise SystemExit(f"expected one {base.__name__} subclass in {root}/app.py, got {found}")
        return found[0]

    server_cls = defined_here(SimpleResourcesServer)
    config_cls = defined_here(BaseResourcesServerConfig)
    config = config_cls(host=args.host, port=args.port, entrypoint="app.py", name=args.server)
    server = server_cls(config=config, server_client=MagicMock(spec=ServerClient))
    app = server.setup_webserver()
    print(f"serving {args.server} on http://{args.host}:{args.port}  (data: {root / 'data'})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
