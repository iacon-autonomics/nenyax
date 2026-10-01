"""Where a run's outputs go: the stack's storage plug.

    store = storage.from_stack(config.get("stack"), client=client, run_id=run_id)
    uri = store.put("/tmp/sft.jsonl", "trajectories-sft.jsonl", kind="dataset")

Plugs: ``nenyax`` (upload to the platform), ``s3``, ``r2`` (S3-compatible), ``gcs`` and ``hf``
(Hugging Face Hub). Credentials arrive as env vars from your connections:

* S3: ``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY``; the platform stores one secret as
  ``KEY_ID:SECRET`` in ``AWS_SECRET_ACCESS_KEY``, which is split here.
* R2: ``R2_SECRET_ACCESS_KEY`` (``KEY_ID:SECRET``) and an endpoint URL in the plug config.
* GCS: ``GOOGLE_APPLICATION_CREDENTIALS_JSON`` (a service-account JSON) or ambient credentials.
* HF: ``HF_TOKEN``.

The location comes from the plug's config: ``{"uri": "s3://bucket/prefix"}``,
``{"uri": "gs://bucket/prefix"}``, ``{"repo": "org/name", "repo_type": "dataset"}``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KINDS = ("checkpoint", "adapter", "prompt", "dataset", "report", "log", "other")


class StorageError(RuntimeError):
    pass


@dataclass
class Stored:
    uri: str
    bytes: int
    sha256: str


def digest(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return path.stat().st_size, h.hexdigest()


def _split_key(var: str, id_var: str) -> tuple[str | None, str | None]:
    """``KEY_ID:SECRET`` in one variable (how the platform stores it), or two variables."""
    secret = os.environ.get(var)
    key_id = os.environ.get(id_var)
    if secret and ":" in secret and not key_id:
        key_id, secret = secret.split(":", 1)
    return key_id, secret


def _bucket(uri: str, scheme: str) -> tuple[str, str]:
    if not uri.startswith(f"{scheme}://"):
        raise StorageError(f"expected a {scheme}:// location, got {uri!r}")
    bucket, _, prefix = uri.removeprefix(f"{scheme}://").partition("/")
    return bucket, prefix.strip("/")


class Storage(ABC):
    plug: str

    @abstractmethod
    def put(self, local: str | Path, name: str, kind: str = "other") -> Stored: ...


class NenyaxStorage(Storage):
    """Upload to the platform, which hosts the file and serves downloads (``nenyax://``)."""

    plug = "nenyax"

    def __init__(self, url: str, token: str, run_id: int) -> None:
        self.url, self.token, self.run_id = url.rstrip("/"), token, run_id

    def put(self, local, name, kind="other") -> Stored:
        path = Path(local)
        size, sha = digest(path)
        return Stored(uri=f"nenyax://upload/{self.run_id}/{name}", bytes=size, sha256=sha)


class S3Storage(Storage):
    plug = "s3"

    def __init__(self, uri: str, endpoint: str | None = None, *, key_vars=None) -> None:
        self.bucket, self.prefix = _bucket(uri, "s3")
        self.endpoint = endpoint
        self.key_vars = key_vars or ("AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID")

    def client(self) -> Any:
        try:
            import boto3
        except ImportError as e:
            raise StorageError("pip install boto3 to store artifacts in S3 or R2") from e
        key_id, secret = _split_key(*self.key_vars)
        kwargs: dict[str, Any] = {}
        if key_id and secret:
            kwargs.update(aws_access_key_id=key_id, aws_secret_access_key=secret)
        if self.endpoint:
            kwargs["endpoint_url"] = self.endpoint
        return boto3.client("s3", **kwargs)

    def put(self, local, name, kind="other") -> Stored:
        path = Path(local)
        size, sha = digest(path)
        key = "/".join(p for p in (self.prefix, name) if p)
        self.client().upload_file(str(path), self.bucket, key)
        return Stored(uri=f"s3://{self.bucket}/{key}", bytes=size, sha256=sha)


class R2Storage(S3Storage):
    """Cloudflare R2 speaks S3; the account endpoint goes in the plug config."""

    plug = "r2"

    def __init__(self, uri: str, endpoint: str | None) -> None:
        if not endpoint:
            raise StorageError(
                "R2 needs its endpoint URL (https://<account>.r2.cloudflarestorage.com)"
            )
        super().__init__(
            uri.replace("r2://", "s3://", 1),
            endpoint,
            key_vars=("R2_SECRET_ACCESS_KEY", "R2_ACCESS_KEY_ID"),
        )

    def put(self, local, name, kind="other") -> Stored:
        stored = super().put(local, name, kind)
        stored.uri = stored.uri.replace("s3://", "r2://", 1)
        return stored


class GCSStorage(Storage):
    plug = "gcs"

    def __init__(self, uri: str) -> None:
        self.bucket, self.prefix = _bucket(uri, "gs")

    def put(self, local, name, kind="other") -> Stored:
        try:
            from google.cloud import storage as gcs
        except ImportError as e:
            raise StorageError("pip install google-cloud-storage to store artifacts in GCS") from e
        path = Path(local)
        size, sha = digest(path)
        raw = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        client = gcs.Client.from_service_account_info(json.loads(raw)) if raw else gcs.Client()
        key = "/".join(p for p in (self.prefix, name) if p)
        client.bucket(self.bucket).blob(key).upload_from_filename(str(path))
        return Stored(uri=f"gs://{self.bucket}/{key}", bytes=size, sha256=sha)


class HFStorage(Storage):
    plug = "hf"

    def __init__(self, repo: str, repo_type: str = "dataset", prefix: str = "") -> None:
        self.repo, self.repo_type, self.prefix = repo, repo_type, prefix.strip("/")

    def put(self, local, name, kind="other") -> Stored:
        try:
            from huggingface_hub import HfApi
        except ImportError as e:
            raise StorageError("pip install huggingface_hub to store artifacts on the Hub") from e
        path = Path(local)
        size, sha = digest(path)
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(self.repo, repo_type=self.repo_type, exist_ok=True, private=True)
        target = "/".join(p for p in (self.prefix, name) if p)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=target,
            repo_id=self.repo,
            repo_type=self.repo_type,
        )
        return Stored(uri=f"hf://{self.repo_type}s/{self.repo}/{target}", bytes=size, sha256=sha)


def from_stack(
    stack: dict[str, Any] | None,
    *,
    url: str | None = None,
    token: str | None = None,
    run_id: int | None = None,
) -> Storage | None:
    """The run's storage plug, or the platform when there is none and the runner is connected."""
    slot = dict((stack or {}).get("storage") or {})
    plug = slot.get("plug") or "nenyax"
    cfg = dict(slot.get("config") or {})
    location = cfg.get("uri") or cfg.get("base_url") or slot.get("base_url")
    if plug == "nenyax":
        if not (url and token and run_id is not None):
            return None
        return NenyaxStorage(url, token, run_id)
    if plug == "s3":
        return S3Storage(_need(location, "s3://bucket/prefix"), cfg.get("endpoint"))
    if plug == "r2":
        return R2Storage(_need(location, "r2://bucket/prefix"), cfg.get("endpoint"))
    if plug == "gcs":
        return GCSStorage(_need(location, "gs://bucket/prefix"))
    if plug == "hf":
        repo = cfg.get("repo") or (location or "").removeprefix("hf://")
        return HFStorage(
            _need(repo, "org/name"), cfg.get("repo_type", "dataset"), cfg.get("prefix", "")
        )
    raise StorageError(f"unknown storage plug {plug!r}")


def _need(value: str | None, example: str) -> str:
    if not value:
        raise StorageError(f"the storage plug needs a location, e.g. {example}")
    return value


# -- reporting artifacts to the platform ---------------------------------------------------------


def report(
    url: str,
    token: str,
    run_id: int,
    *,
    kind: str,
    name: str,
    stored: Stored,
    meta: dict[str, Any] | None = None,
    file: Path | None = None,
) -> dict[str, Any]:
    """Record an artifact on the platform; with ``file``, upload it too (the ``nenyax`` plug)."""
    fields = {
        "kind": kind if kind in KINDS else "other",
        "name": name,
        "uri": stored.uri,
        "bytes": str(stored.bytes),
        "sha256": stored.sha256,
        "meta": json.dumps(meta or {}),
    }
    boundary = uuid.uuid4().hex
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        for k, v in fields.items()
    ]
    if file is not None:
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
                "Content-Type: application/octet-stream\r\n\r\n"
            ).encode()
            + file.read_bytes()
            + b"\r\n"
        )
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/runner/runs/{run_id}/artifacts",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read() or b"{}")


def publish(
    items: list[dict[str, Any]],
    store: Storage | None,
    *,
    url: str | None,
    token: str | None,
    run_id: int | None,
) -> list[dict[str, Any]]:
    """Store each produced file with the storage plug and record it on the platform."""
    out = []
    for item in items:
        path = Path(item["path"])
        if not path.exists() or store is None:
            continue
        stored = store.put(path, item["name"], item.get("kind", "other"))
        record = {"kind": item.get("kind", "other"), "name": item["name"], "uri": stored.uri}
        if url and token and run_id is not None:
            upload = path if isinstance(store, NenyaxStorage) else None
            record = report(
                url,
                token,
                run_id,
                kind=record["kind"],
                name=item["name"],
                stored=stored,
                meta=item.get("meta"),
                file=upload,
            )
        out.append(record)
    return out


def scratch() -> Path:
    return Path(tempfile.mkdtemp(prefix="nenyax-artifacts-"))
