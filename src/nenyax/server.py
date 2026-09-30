"""A recording, OpenAI-compatible model server.

Rollout- and task-mode environments call a model endpoint themselves. To stay format-agnostic,
Nenyax always puts this server in between:

* backed by a :class:`~nenyax.policy.ChatModel` (e.g. a Python function), it *is* the model;
* backed by an :class:`~nenyax.policy.Endpoint`, it transparently proxies to that endpoint.

Either way it records every request/response as a :class:`~nenyax.types.ModelCall`, including
token ids and logprobs when the upstream returns them. Calls are grouped by a per-episode path
prefix (``/s/<session>/v1``) so concurrent episodes never mix.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .policy import ChatModel, Endpoint
from .types import Message, ModelCall, extract_tokens

_DEFAULT_SESSION = "_"


class ModelServer:
    def __init__(
        self,
        backend: ChatModel | Endpoint,
        *,
        model_name: str | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.backend = backend
        self.model_name = model_name or getattr(backend, "model", None) or "nenyax-policy"
        self._calls: dict[str, list[ModelCall]] = defaultdict(list)
        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer((host, port), self._make_handler())
        self._httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------------------------

    def start(self) -> ModelServer:
        if self._thread is None:
            self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread = None

    def __enter__(self) -> ModelServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- addressing and recording ------------------------------------------------------------

    @property
    def host(self) -> str:
        return self._httpd.server_address[0]

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    def base_url(self, session: str | None = None, *, host: str | None = None) -> str:
        """OpenAI-style base URL (ending in ``/v1``) whose calls are recorded under ``session``.

        ``host`` overrides the advertised hostname, e.g. ``host.docker.internal`` for containers.
        """
        root = f"http://{host or self.host}:{self.port}"
        return f"{root}/s/{session}/v1" if session else f"{root}/v1"

    def new_session(self) -> str:
        return uuid.uuid4().hex[:12]

    def calls(self, session: str | None = None) -> list[ModelCall]:
        with self._lock:
            return list(self._calls.get(session or _DEFAULT_SESSION, []))

    def pop_calls(self, session: str | None = None) -> list[ModelCall]:
        with self._lock:
            return self._calls.pop(session or _DEFAULT_SESSION, [])

    def _record(self, session: str, call: ModelCall) -> None:
        with self._lock:
            self._calls[session].append(call)

    # -- model logic -------------------------------------------------------------------------

    def _chat(self, body: dict[str, Any]) -> dict[str, Any]:
        if isinstance(self.backend, Endpoint):
            upstream = dict(body)
            upstream["model"] = self.backend.model
            upstream.pop("stream", None)
            upstream.pop("stream_options", None)
            return self.backend.chat(upstream)

        messages = [Message.model_validate(m) for m in body.get("messages", [])]
        params = {k: v for k, v in body.items() if k not in ("messages", "model", "stream")}
        reply = self.backend.complete(messages, **params)
        message = reply.to_openai()
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model") or self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if reply.tool_calls else "stop",
                    "logprobs": None,
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    # -- HTTP plumbing -----------------------------------------------------------------------

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # keep test output clean
                pass

            def _route(self) -> tuple[str, str]:
                path = self.path.split("?", 1)[0]
                session = _DEFAULT_SESSION
                if path.startswith("/s/"):
                    _, _, session, path = path.split("/", 3)
                    path = "/" + path
                return session, path

            def _send(self, code: int, payload: dict[str, Any]) -> None:
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_stream(self, response: dict[str, Any]) -> None:
                choice = response["choices"][0]
                delta = dict(choice["message"])
                if delta.get("tool_calls"):  # streamed tool calls carry an index
                    delta["tool_calls"] = [
                        {"index": i, **tc} for i, tc in enumerate(delta["tool_calls"])
                    ]
                chunk = {
                    "id": response["id"],
                    "object": "chat.completion.chunk",
                    "created": response["created"],
                    "model": response["model"],
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                final = {
                    **chunk,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}
                    ],
                }
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                for part in (chunk, final):
                    self.wfile.write(f"data: {json.dumps(part)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True

            def do_GET(self) -> None:
                _, path = self._route()
                if path.rstrip("/") in ("/v1/models", "/models"):
                    self._send(
                        200,
                        {
                            "object": "list",
                            "data": [
                                {"id": server.model_name, "object": "model", "owned_by": "nenyax"}
                            ],
                        },
                    )
                elif path.rstrip("/") in ("/health", "/v1/health"):
                    self._send(200, {"status": "ok"})
                else:
                    self._send(404, {"error": {"message": f"unknown path {path}"}})

            def do_POST(self) -> None:
                session, path = self._route()
                if not path.rstrip("/").endswith("/chat/completions"):
                    self._send(404, {"error": {"message": f"unsupported path {path}"}})
                    return
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                call = ModelCall(request=body)
                t0 = time.perf_counter()
                try:
                    response = server._chat(body)
                except Exception as e:  # report to the caller and keep the record
                    call.error = f"{type(e).__name__}: {e}"
                    call.latency_s = time.perf_counter() - t0
                    server._record(session, call)
                    self._send(500, {"error": {"message": call.error, "type": "server_error"}})
                    return
                call.response = response
                call.latency_s = time.perf_counter() - t0
                extract_tokens(call, response)
                server._record(session, call)
                if body.get("stream"):
                    self._send_stream(response)
                else:
                    self._send(200, response)

        return Handler
