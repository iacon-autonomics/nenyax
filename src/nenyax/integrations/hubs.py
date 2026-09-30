"""Environment hubs: find environments where they already live, get back loadable Nenyax URIs.

    for hit in nenyax.integrations.search("prime", "math", limit=5):
        print(hit.uri, hit.description)
    env = nenyax.load(nenyax.integrations.fetch(hit))       # installs/downloads if needed

Built in: Hugging Face (OpenEnv Spaces), the Prime Intellect Environments Hub and the Harbor
registry. All three are public and need no key to search. More hubs register through the
``nenyax.hubs`` entry-point group.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from abc import abstractmethod
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from ..environment import NenyaxError
from .base import Integration

CACHE = Path(os.environ.get("NENYAX_CACHE", Path.home() / ".cache" / "nenyax"))


def _get_json(url: str, timeout: float = 30.0, token: str | None = None) -> Any:
    headers = {"User-Agent": "nenyax", **({"Authorization": f"Bearer {token}"} if token else {})}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
        return json.loads(r.read())


@dataclass(frozen=True)
class Listing:
    hub: str
    name: str
    uri: str
    """Nenyax URI to load once fetched."""
    description: str = ""
    meta: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)


class Hub(Integration):
    kind = "hub"

    @abstractmethod
    def search(self, query: str = "", limit: int = 20) -> list[Listing]: ...

    def fetch(self, listing: Listing) -> str:
        """Make ``listing`` loadable locally (install, download); returns its URI."""
        return listing.uri


class HuggingFaceHub(Hub):
    """OpenEnv environments hosted as Hugging Face Spaces (tag ``openenv``)."""

    name = "huggingface"
    summary = "OpenEnv environments on Hugging Face Spaces"
    verified_live = True

    def search(self, query: str = "", limit: int = 20) -> list[Listing]:
        params = [("filter", "openenv"), ("limit", str(limit))]
        params += [("expand[]", f) for f in ("subdomain", "runtime", "likes", "tags")]
        if query:
            params.append(("search", query))
        url = "https://huggingface.co/api/spaces?" + urllib.parse.urlencode(params)
        rows = _get_json(url, token=self.credential("HF_TOKEN"))
        listings = []
        for r in rows:
            if not r.get("subdomain"):  # never built or not servable
                continue
            stage = (r.get("runtime") or {}).get("stage")
            listings.append(
                Listing(
                    self.name,
                    r["id"],
                    f"openenv:https://{r['subdomain']}.hf.space",
                    meta={"stage": stage, "likes": r.get("likes"), "tags": r.get("tags", [])},
                )
            )
        return listings


class PrimeHub(Hub):
    """The Prime Intellect Environments Hub (Verifiers environments, installed as packages)."""

    name = "prime"
    summary = "Prime Intellect Environments Hub (Verifiers)"
    verified_live = True
    API = "https://api.primeintellect.ai/api/v1/environmentshub/"

    def search(self, query: str = "", limit: int = 20) -> list[Listing]:
        params = {"limit": str(limit), **({"search": query} if query else {})}
        data = _get_json(self.API + "?" + urllib.parse.urlencode(params))["data"]
        return [
            Listing(
                self.name,
                f"{row['owner']['name']}/{row['name']}",
                f"verifiers:{row['name'].replace('-', '_')}",
                (row.get("description") or "").strip(),
                meta={"visibility": row.get("visibility"), "updated_at": row.get("updated_at")},
            )
            for row in data
        ]

    def fetch(self, listing: Listing) -> str:
        """Install the environment package (public environments need no login)."""
        cmd = [sys.executable, "-m", "verifiers.cli.commands.install", listing.name]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise NenyaxError(f"installing {listing.name} failed: {proc.stderr[-500:]}")
        return listing.uri


class HarborRegistry(Hub):
    """Harbor's dataset registry: Terminal-Bench and 80+ adapted benchmarks."""

    name = "harbor"
    summary = "Harbor registry (Terminal-Bench, SWE-bench adapters, ...)"
    verified_live = True
    REGISTRY = "https://raw.githubusercontent.com/laude-institute/harbor/main/registry.json"

    def _registry(self) -> list[dict[str, Any]]:
        return _get_json(self.REGISTRY, timeout=60)

    def search(self, query: str = "", limit: int = 20) -> list[Listing]:
        q = query.lower()
        hits = [
            d
            for d in self._registry()
            if q in d["name"].lower() or q in (d.get("description") or "").lower()
        ]
        return [
            Listing(
                self.name,
                f"{d['name']}@{d['version']}",
                f"harbor:{CACHE / 'harbor' / (d['name'] + '@' + d['version'])}",
                (d.get("description") or "").strip(),
                meta={"tasks": d["tasks"]},
            )
            for d in hits[:limit]
        ]

    def fetch(self, listing: Listing, *, max_tasks: int | None = None) -> str:
        """Sparse-checkout each task directory at its pinned commit into the Nenyax cache."""
        root = Path(listing.uri.removeprefix("harbor:"))
        for task in listing.meta["tasks"][:max_tasks]:
            dest = root / task["name"]
            if (dest / "task.toml").is_file():
                continue
            _sparse_checkout(task["git_url"], task["git_commit_id"], task["path"], dest)
        return listing.uri


def _sparse_checkout(git_url: str, commit: str, path: str, dest: Path) -> None:
    # One sparse clone per (repo, commit), so datasets pinned to different commits never clash.
    repos = CACHE / "git" / f"{urllib.parse.quote(git_url, safe='')}@{commit[:12]}"
    if not (repos / ".git").is_dir():
        repos.mkdir(parents=True, exist_ok=True)
        _git(repos, "init", "-q")
        _git(repos, "remote", "add", "origin", git_url)
        _git(repos, "sparse-checkout", "init", "--cone")
    _git(repos, "sparse-checkout", "add", path)
    _git(repos, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit)
    _git(repos, "checkout", "-q", commit)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(repos / path, dest, dirs_exist_ok=True)


def _git(cwd: Path, *args: str) -> None:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise NenyaxError(f"git {' '.join(args)} failed: {proc.stderr.strip()[:300]}")


_BUILTIN_HUBS = (HuggingFaceHub, PrimeHub, HarborRegistry)


def hubs() -> dict[str, Hub]:
    found: dict[str, Hub] = {cls.name: cls() for cls in _BUILTIN_HUBS}
    for ep in entry_points(group="nenyax.hubs"):
        hub = ep.load()()
        found[hub.name] = hub
    return found


def search(hub: str, query: str = "", limit: int = 20) -> list[Listing]:
    return hubs()[hub].search(query, limit)


def fetch(listing: Listing, **options: Any) -> str:
    return hubs()[listing.hub].fetch(listing, **options)
