"""The Nenyax platform from the command line: sign in, create an environment, push it.

    nenyax login --url https://<platform>/api --token nxp_...
    nenyax new-env my-env          # a folder with nenyax.toml, env.py, README.md
    nenyax push my-env [--public]  # upload a new version; the platform hosts and checks it

Pushed environments run anywhere a runner does: ``nenyax runner`` and Nenyax Cloud fetch the
package, install its requirements and load its entrypoint (see :func:`load_package`).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tomllib
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

CONFIG = Path(os.environ.get("NENYAX_CONFIG_DIR", Path.home() / ".config" / "nenyax"))
CACHE = Path(os.environ.get("NENYAX_CACHE", Path.home() / ".cache" / "nenyax")) / "packages"
SKIP = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache"}


class PlatformError(RuntimeError):
    pass


# -- credentials ---------------------------------------------------------------------------------


def save_login(url: str, token: str) -> Path:
    CONFIG.mkdir(parents=True, exist_ok=True)
    path = CONFIG / "credentials.json"
    path.write_text(json.dumps({"url": url.rstrip("/"), "token": token}, indent=2))
    path.chmod(0o600)
    return path


def credentials() -> tuple[str, str]:
    url, token = os.environ.get("NENYAX_PLATFORM_URL"), os.environ.get("NENYAX_TOKEN")
    path = CONFIG / "credentials.json"
    if (not url or not token) and path.exists():
        saved = json.loads(path.read_text())
        url, token = url or saved.get("url"), token or saved.get("token")
    if not url or not token:
        raise PlatformError("not signed in: run `nenyax login --url <url>/api --token nxp_...`")
    return url.rstrip("/"), token


def _request(
    url: str, token: str, *, data: bytes | None = None, headers=None, method: str | None = None
) -> bytes:
    auth = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(url, data=data, headers={**auth, **(headers or {})}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        with contextlib.suppress(ValueError):
            body = json.loads(body).get("detail", body)
        raise PlatformError(f"{e.code}: {body}") from None


def whoami() -> dict[str, Any]:
    url, token = credentials()
    return json.loads(_request(f"{url}/auth/me", token) or b"null") or {}


# -- creating and pushing ------------------------------------------------------------------------

ENV_PY = '''"""{title}: a Nenyax environment.

Edit TASKS and score(), then `nenyax try .` to play it and `nenyax push .` to publish.
"""

import nenyax

TASKS = [
    {{"prompt": "What is 17 + 25? Answer with just the number.", "answer": "42"}},
    {{"prompt": "What is 9 * 8? Answer with just the number.", "answer": "72"}},
]


def score(task, reply):
    return float(reply.strip().split()[-1:] == [task["answer"]]) if reply.strip() else 0.0


env = nenyax.define("{name}", tasks=TASKS, score=score)
'''

TOML = """[environment]
name = "{name}"
entrypoint = "env:env"        # module:attr, an Environment or a function returning one
version = "0.1.0"
description = "Describe what this environment tests and how it is scored."
license = "MIT"
modality = "text"
tags = []
requirements = []             # pip specs the environment needs, installed on the runner
"""

README = """# {title}

What the environment tests, where its tasks come from, and how a reply is scored.

## Try it

```bash
nenyax try .
```
"""


def new_env(name: str, dest: str | Path = ".") -> Path:
    root = Path(dest) / name
    if root.exists():
        raise PlatformError(f"{root} already exists")
    root.mkdir(parents=True)
    title = name.replace("-", " ").replace("_", " ").title()
    (root / "env.py").write_text(ENV_PY.format(name=name, title=title))
    (root / "nenyax.toml").write_text(TOML.format(name=name))
    (root / "README.md").write_text(README.format(title=title))
    return root


def read_manifest(path: str | Path) -> dict[str, Any]:
    toml = Path(path) / "nenyax.toml"
    if not toml.exists():
        raise PlatformError(f"no nenyax.toml in {path} (create one with `nenyax new-env`)")
    meta = tomllib.loads(toml.read_text()).get("environment")
    if not meta or "name" not in meta or "entrypoint" not in meta:
        raise PlatformError("nenyax.toml needs [environment] with name and entrypoint")
    return meta


def pack(path: str | Path) -> bytes:
    """A deterministic .tar.gz of the folder: same files, same bytes (no timestamps, no owners)."""
    import gzip

    root = Path(path).resolve()
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for file in sorted(root.rglob("*")):
            rel = file.relative_to(root)
            if any(part in SKIP or part.endswith(".pyc") for part in rel.parts):
                continue
            if file.is_file() and not file.is_symlink():
                data = file.read_bytes()
                info = tarfile.TarInfo(str(rel))
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                tar.addfile(info, io.BytesIO(data))
    return gzip.compress(raw.getvalue(), mtime=0)


def push(path: str | Path, *, public: bool = False, org: str | None = None) -> dict[str, Any]:
    """Upload the folder at ``path`` as a new version of its environment."""
    meta = read_manifest(path)
    if public:
        meta["visibility"] = "public"
    if org:
        meta["org"] = org
    blob = pack(path)
    url, token = credentials()
    boundary = uuid.uuid4().hex
    body = b"".join(
        [
            f'--{boundary}\r\nContent-Disposition: form-data; name="manifest"\r\n\r\n'.encode(),
            json.dumps(meta).encode(),
            f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="package"; '
            f'filename="{meta["name"]}.tar.gz"\r\nContent-Type: application/gzip\r\n\r\n'.encode(),
            blob,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    out = _request(
        f"{url}/packages",
        token,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    return {**json.loads(out), "bytes": len(blob)}


# -- loading on a runner -------------------------------------------------------------------------


def _import_from(root: Path, entrypoint: str) -> Any:
    """Import ``module:attr`` from ``root``. Environment folders all tend to call their module
    ``env``; a copy cached from another folder must not answer for this one."""
    import importlib

    top = entrypoint.split(":")[0].split(".")[0]
    cached = sys.modules.get(top)
    where = getattr(cached, "__file__", None) or ""
    if cached is not None and not Path(where).resolve().is_relative_to(root.resolve()):
        for name in [m for m in sys.modules if m == top or m.startswith(f"{top}.")]:
            del sys.modules[name]
        importlib.invalidate_caches()
    if str(root) in sys.path:  # this folder first, so its module wins the lookup
        sys.path.remove(str(root))
    sys.path.insert(0, str(root))
    from .config import _import

    return _import(entrypoint)


def load_package(fetch: dict[str, Any], url: str, token: str) -> Any:
    """Download a pushed environment, install its requirements, and load its entrypoint."""
    from .environment import Environment

    slug, version = fetch["slug"], fetch["version"]
    home = CACHE / slug / version
    if not (home / ".ready").exists():
        blob = _request(f"{url}/packages/{slug}/download?version={version}", token)
        home.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            tar.extractall(home, filter="data")
        if reqs := fetch.get("requirements"):
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", "--target", str(home / "_deps")]
                + list(reqs),
                capture_output=True,
                text=True,
                timeout=900,
            )
            if proc.returncode != 0:
                raise PlatformError(f"installing {reqs} failed:\n{proc.stderr[-2000:]}")
        (home / ".ready").touch()
    for p in (home / "_deps", home):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    obj = _import_from(home, fetch["entrypoint"])
    return obj if isinstance(obj, Environment) else obj()


def load_folder(path: str | Path) -> Any:
    """Load a local environment folder by its ``nenyax.toml`` entrypoint (before pushing it)."""
    from .config import _import
    from .environment import Environment

    root = Path(path).resolve()
    meta = read_manifest(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    obj = _import_from(root, meta["entrypoint"])
    return obj if isinstance(obj, Environment) else obj()
