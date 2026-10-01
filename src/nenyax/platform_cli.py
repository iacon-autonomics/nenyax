"""The platform from the terminal, like ``gh``: everything the web app does, as commands.

    nenyax auth login                      # browser sign-in (or --with-token for CI)
    nenyax env push ./my-env --public      # publish; Nenyax Cloud checks it
    nenyax project create arith --env reasoning-gym/basic_arithmetic --model openai/gpt-4o-mini
    nenyax project run arith --watch       # run the pipeline on Nenyax Cloud and stream it
    nenyax connection add openai           # store a provider key (prompted, never echoed)
    nenyax api GET /projects               # anything else, raw

Every command takes ``--json`` for scripts.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
import time
import webbrowser
from typing import Any

from . import platform as pf

DEFAULT_URL = "http://127.0.0.1:4300/api"


# -- plumbing ------------------------------------------------------------------------------------


def call(method: str, path: str, body: Any = None, *, auth: bool = True) -> Any:
    url, token = pf.credentials() if auth else (_url(), "")
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if data is not None else {}
    out = pf._request(f"{url}{path}", token, data=data, headers=headers, method=method)
    return json.loads(out) if out else None


def _url() -> str:
    try:
        return pf.credentials()[0]
    except pf.PlatformError:
        return DEFAULT_URL


def _emit(args: argparse.Namespace, data: Any, human) -> None:
    if getattr(args, "json", False):
        print(json.dumps(data, indent=2, default=str))
    else:
        human(data)


def _table(rows: list[list[Any]], headers: list[str]) -> None:
    if not rows:
        print("  (none)")
        return
    cells = [headers, *[[("" if c is None else str(c)) for c in r] for r in rows]]
    widths = [min(max(len(r[i]) for r in cells), 60) for i in range(len(headers))]
    for i, row in enumerate(cells):
        print("  ".join(c[: widths[j]].ljust(widths[j]) for j, c in enumerate(row)).rstrip())
        if i == 0:
            print("  ".join("─" * w for w in widths))


def _me() -> dict[str, Any]:
    return call("GET", "/auth/me") or {}


def _slug(name: str) -> str:
    """``owner/name``; a bare name means one of yours."""
    return name if "/" in name else f"{_me().get('handle')}/{name}"


def _score(x: Any) -> str:
    return f"{x:.3f}" if isinstance(x, int | float) else "—"


def _params(pairs: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        try:
            out[key] = json.loads(raw)
        except ValueError:
            out[key] = raw
    return out


def _web(path: str) -> str:
    return _url().removesuffix("/api") + path


# -- auth ----------------------------------------------------------------------------------------


def auth_login(args: argparse.Namespace) -> None:
    url = args.url.rstrip("/")
    if args.with_token:
        token = sys.stdin.readline().strip() if args.with_token == "-" else args.with_token
        pf.save_login(url, token)
        me = pf.whoami()
        print(f"✓ signed in as {me.get('name')} (@{me.get('handle')})")
        return
    start = json.loads(
        pf._request(
            f"{url}/auth/device",
            "",
            data=json.dumps({"client_name": args.name}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    )
    print(f"! First copy your one-time code: {start['user_code']}", flush=True)
    print(f"  Then approve it at {start['verification_url']}", flush=True)
    if not args.no_browser:
        webbrowser.open(start["verification_url"])
    deadline = time.time() + start["expires_in"]
    while time.time() < deadline:
        time.sleep(start.get("interval", 2))
        reply = json.loads(
            pf._request(
                f"{url}/auth/device/token",
                "",
                data=json.dumps({"device_code": start["device_code"]}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        )
        if reply.get("status") == "approved":
            pf.save_login(url, reply["token"])
            user = reply["user"]
            print(f"✓ signed in as {user['name']} (@{user['handle']}) at {url}")
            return
    sys.exit("timed out waiting for approval; run `nenyax auth login` again")


def auth_status(args: argparse.Namespace) -> None:
    try:
        url, token = pf.credentials()
        me = _me()
    except pf.PlatformError as e:
        sys.exit(str(e))
    _emit(
        args,
        {"url": url, "user": me, "token": f"nxp_…{token[-4:]}"},
        lambda d: print(
            f"✓ {d['url']}\n  signed in as {me.get('name')} (@{me.get('handle')}), "
            f"{me.get('email')}\n  token {d['token']}"
        ),
    )


def auth_logout(args: argparse.Namespace) -> None:
    path = pf.CONFIG / "credentials.json"
    if path.exists():
        path.unlink()
    print("signed out (the token still works until you revoke it in Settings)")


def auth_token(args: argparse.Namespace) -> None:
    print(pf.credentials()[1])


# -- environments --------------------------------------------------------------------------------


def env_list(args: argparse.Namespace) -> None:
    from urllib.parse import urlencode

    q = {"q": args.query or None, "kind": args.kind, "source": args.source, "limit": args.limit}
    if args.mine:
        q["mine"] = "true"
    data = call("GET", "/listings?" + urlencode({k: v for k, v in q.items() if v}))
    _emit(
        args,
        data,
        lambda d: (
            _table(
                [
                    [i["slug"], i["kind"], i.get("tier") or "", i.get("stars", 0)]
                    for i in d["items"]
                ],
                ["SLUG", "KIND", "TIER", "★"],
            ),
            print(f"\n  {len(d['items'])} of {d['total']:,}"),
        ),
    )


def env_view(args: argparse.Namespace) -> None:
    if args.web:
        webbrowser.open(_web(f"/l/{args.slug}"))
        return
    d = call("GET", f"/listings/{args.slug}")

    def human(d):
        print(f"{d['name']}  ({d['slug']})")
        print(
            f"  {d['kind']} · {d['source']} · tier {d.get('tier') or 'not checked'} · ★ {d.get('stars', 0)}"
        )
        if d.get("summary"):
            print(f"  {d['summary']}")
        if d.get("uri"):
            print(f"  uri      {d['uri']}")
        if d.get("versions"):
            print(f"  version  {d['versions'][0]['version']}")
        if args.readme and d.get("readme"):
            print("\n" + d["readme"])

    _emit(args, d, human)


def env_push(args: argparse.Namespace) -> None:
    try:
        out = pf.push(args.path, public=args.public, org=args.org)
    except pf.PlatformError as e:
        sys.exit(str(e))

    def human(out):
        print(f"✓ {'published' if out['created'] else 'updated'} {out['slug']} {out['version']}")
        print(f"  {out['url']}")
        if out.get("check_run"):
            print(f"  conformance check on Nenyax Cloud: nenyax runs watch {out['check_run']}")

    _emit(args, out, human)


def env_new(args: argparse.Namespace) -> None:
    root = pf.new_env(args.name, args.dest)
    print(f"✓ created {root}/\n  try it:  nenyax try {root}\n  push it: nenyax env push {root}")


def env_star(args: argparse.Namespace) -> None:
    d = call("DELETE" if args.undo else "POST", f"/listings/{args.slug}/star")
    print(f"{'☆ unstarred' if args.undo else '★ starred'} {args.slug} ({d.get('stars')} stars)")


def env_run(args: argparse.Namespace, kind: str) -> None:
    model = _model(args.model) if getattr(args, "model", None) else None
    body = {"kind": kind, "listing": args.slug, "model": model, "params": _params(args.param)}
    run = call("POST", "/runs", body)
    print(f"✓ queued {kind} run {run['id']} on {run['config'].get('target', 'cloud')}")
    if args.watch:
        _watch(run["id"])


# -- projects ------------------------------------------------------------------------------------


def _model(spec: str) -> dict[str, Any]:
    """``provider/model`` (e.g. openai/gpt-4o-mini), wired to your connection for that provider."""
    provider, _, name = spec.partition("/")
    if not name:
        sys.exit("--model is provider/model, e.g. openai/gpt-4o-mini or vllm/Qwen/Qwen3-8B")
    model: dict[str, Any] = {"provider": provider, "name": name}
    for c in call("GET", "/connections") or []:
        if c["service"] == provider:
            model["connection_id"] = c["id"]
            break
    return model


def project_list(args: argparse.Namespace) -> None:
    data = call("GET", "/projects")
    _emit(
        args,
        data,
        lambda d: _table(
            [
                [
                    p["slug"],
                    (p.get("environment") or {}).get("slug", ""),
                    _score(p.get("baseline")),
                    _score(p.get("best")),
                    (p.get("last_run") or {}).get("status", "never run"),
                ]
                for p in d
            ],
            ["PROJECT", "ENVIRONMENT", "BASELINE", "BEST", "LAST RUN"],
        ),
    )


def project_create(args: argparse.Namespace) -> None:
    body: dict[str, Any] = {
        "name": args.name,
        "goal": args.goal,
        "visibility": "public" if args.public else "private",
    }
    if args.env:
        body["environment" if "/" in args.env and ":" not in args.env else "environment_uri"] = (
            args.env
        )
    if args.model:
        body["model"] = _model(args.model)
    if args.learner:
        body["learner"] = {
            "use": args.learner,
            **({"install": args.install} if args.install else {}),
        }
    if args.org:
        body["org"] = args.org
    p = call("POST", "/projects", body)
    if args.runners:
        call("PATCH", f"/projects/{p['slug']}", {"compute": {"target": "runners"}})
    print(
        f"✓ created {p['slug']}\n  next: {p['next_step'].replace('_', ' ')} · {_web('/projects/' + p['slug'])}"
    )
    print(f"  run it: nenyax project run {p['slug']} --watch")


def project_view(args: argparse.Namespace) -> None:
    slug = _slug(args.project)
    if args.web:
        webbrowser.open(_web(f"/projects/{slug}"))
        return
    p = call("GET", f"/projects/{slug}")

    def human(p):
        env = (p.get("environment") or {}).get("slug", "none")
        model = p.get("model") or {}
        print(f"{p['slug']}  ({p['visibility']})")
        print(f"  environment  {env}")
        print(
            f"  model        {model.get('provider', '')}/{model.get('name', '') if model else 'reference policy'}"
        )
        print(f"  learner      {(p.get('learner') or {}).get('use', 'incontext')}")
        print(
            f"  compute      {'Nenyax Cloud' if p.get('compute_target') == 'cloud' else 'your runners'}"
        )
        print(f"  baseline     {_score(p.get('baseline'))}   best {_score(p.get('best'))}")
        print(f"  pipeline     {' → '.join(s['kind'] for s in p.get('pipeline') or [])}")
        print(
            "  checklist    "
            + "  ".join(("✓ " if c["done"] else "○ ") + c["title"] for c in p["checklist"])
        )
        print(f"  next         {p['next_step'].replace('_', ' ')}")

    _emit(args, p, human)


def project_edit(args: argparse.Namespace) -> None:
    body: dict[str, Any] = {}
    if args.env:
        body["environment" if ":" not in args.env else "environment_uri"] = args.env
    if args.model:
        body["model"] = _model(args.model)
    if args.learner:
        body["learner"] = {
            "use": args.learner,
            **({"install": args.install} if args.install else {}),
        }
    if args.visibility:
        body["visibility"] = args.visibility
    if args.compute:
        body["compute"] = {"target": args.compute}
    if args.description:
        body["description"] = args.description
    if args.pipeline:
        body["pipeline"] = [{"kind": k.strip(), "params": {}} for k in args.pipeline.split(",")]
    if not body:
        sys.exit("nothing to change; see `nenyax project edit --help`")
    p = call("PATCH", f"/projects/{_slug(args.project)}", body)
    print(f"✓ updated {p['slug']} · next: {p['next_step'].replace('_', ' ')}")


def project_run(args: argparse.Namespace) -> None:
    slug = _slug(args.project)
    if args.step:
        run = call(
            "POST", f"/projects/{slug}/runs", {"kind": args.step, "params": _params(args.param)}
        )
        print(f"✓ queued {args.step} run {run['id']} for {slug}")
        if args.watch:
            _watch(run["id"])
        return
    pipe = call("POST", f"/projects/{slug}/pipeline")
    steps = " → ".join(s["kind"] for s in pipe.get("steps") or [])
    print(f"✓ started pipeline {pipe['id']} for {slug}: {steps}")
    if args.watch:
        seen: set[int] = set()
        while True:
            runs = [
                r
                for r in call("GET", f"/projects/{slug}/runs")
                if r.get("pipeline_run_id") == pipe["id"]
            ]
            for r in sorted(runs, key=lambda r: r["id"]):
                if r["id"] not in seen:
                    seen.add(r["id"])
                    print(f"\n── step {r.get('step', 0) + 1}: {r['kind']} (run {r['id']})")
                    _watch(r["id"])
            state = next(
                (x for x in call("GET", f"/projects/{slug}/pipelines") if x["id"] == pipe["id"]), {}
            )
            if state.get("status") in ("succeeded", "failed", "cancelled"):
                print(f"\npipeline {state['status']}")
                return
            time.sleep(2)


def project_export(args: argparse.Namespace) -> None:
    url, token = pf.credentials()
    text = pf._request(f"{url}/projects/{_slug(args.project)}/config.toml", token).decode()
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"✓ wrote {args.output}; run it locally with `nenyax train --config {args.output}`")
    else:
        print(text)


def project_stack(args: argparse.Namespace) -> None:
    """Show or change the project's stack: which plug fills each slot."""
    slug = _slug(args.project)
    patch: dict[str, Any] = {}

    def slot(name: str, plug: str | None, conn: int | None, config: dict[str, Any]) -> None:
        if plug:
            patch[name] = {"plug": plug, "connection_id": conn, "config": config}

    compute_cfg = {
        k: v
        for k, v in {
            "gpu": args.gpu,
            "gpu_count": args.gpu_count,
            "cpu": args.cpu,
            "memory_mb": args.memory,
            "region": args.region,
        }.items()
        if v is not None
    }
    compute_cfg.update(_params(args.compute_config))
    slot("compute", args.compute, args.compute_connection, compute_cfg)
    slot("sandbox", args.sandbox, args.sandbox_connection, {})
    slot("trainer", args.trainer, args.trainer_connection, _params(args.trainer_config))
    slot("storage", args.storage, args.storage_connection, {})
    slot("serving", args.serving, None, {})
    if args.tracking:
        patch["tracking"] = [
            {
                "plug": t.split(":")[0],
                "connection_id": int(t.split(":")[1]) if ":" in t else None,
                "config": {},
            }
            for t in args.tracking
        ]
    routing = {}
    if args.fallback:
        routing["fallback"] = [x.strip() for x in args.fallback.split(",") if x.strip()]
    if args.prefer:
        routing["prefer"] = args.prefer
    if args.max_cost is not None:
        routing["max_cost_usd"] = args.max_cost
    if args.budget is not None:
        routing["project_budget_usd"] = args.budget
    if routing:
        current = (call("GET", f"/projects/{slug}").get("stack") or {}).get("routing") or {}
        patch["routing"] = {**current, **routing}
    p = (
        call("PATCH", f"/projects/{slug}", {"stack": patch})
        if patch
        else call("GET", f"/projects/{slug}")
    )
    stack = p.get("stack") or {}

    def human(stack):
        if patch:
            print(f"✓ updated the stack of {p['slug']}")
        print(f"{p['slug']}  runs on {p.get('compute_target')}")
        for name in ("compute", "sandbox", "trainer", "serving", "storage"):
            entry = stack.get(name) or {}
            if entry:
                conn = (
                    f"  (connection {entry['connection_id']})" if entry.get("connection_id") else ""
                )
                cfg = "  " + json.dumps(entry["config"]) if entry.get("config") else ""
                print(f"  {name:<9} {entry.get('plug')}{conn}{cfg}")
            else:
                print(f"  {name:<9} default")
        for t in stack.get("tracking") or []:
            print(f"  tracking  {t['plug']} (connection {t.get('connection_id')})")
        if stack.get("routing"):
            print(f"  routing   {json.dumps(stack['routing'])}")

    _emit(args, stack, human)


def compute_list(args: argparse.Namespace) -> None:
    d = call("GET", "/compute")

    def human(d):
        conns: dict[str, list] = {}
        for c in d["connections"]:
            conns.setdefault(c["plug"], []).append(c)
        rows = []
        for p in d["plugs"]:
            mine = conns.get(p["id"], [])
            ready = (
                "built in"
                if not p.get("service")
                else (
                    ", ".join(f"#{c['id']} {c['label']} ({c['running']} running)" for c in mine)
                    or f"connect: nenyax connection add {p['service']}"
                )
            )
            rows.append([p["id"], p["status"], "yes" if p.get("gpu") else "", ready])
        _table(rows, ["PLUG", "STATUS", "GPU", "YOUR ACCOUNTS"])

    _emit(args, d, human)


def compute_test(args: argparse.Namespace) -> None:
    d = call("POST", f"/compute/{args.id}/test")
    mark = "✓" if d["ok"] else "✘"
    print(
        f"{mark} {d['plug']} ({d['status']}): about ${d['estimate_usd_hr']}/h for a typical shape"
    )
    for problem in d["problems"]:
        print(f"  {problem}")
    if d["needs"]:
        print(f"  needs: {', '.join(d['needs'])}")


def runs_compare(args: argparse.Namespace) -> None:
    d = call("GET", "/runs/compare?ids=" + ",".join(str(i) for i in args.ids))

    def human(d):
        rows = [
            [
                r["id"],
                r["kind"],
                r["status"],
                r.get("model") or "reference",
                r.get("learner") or "",
                (r.get("provenance") or {}).get("compute") or r["stack"].get("compute", ""),
                _score(r.get("score")),
                f"${r['cost_usd']:.4f}" if isinstance(r.get("cost_usd"), int | float) else "",
                r.get("seconds") or "",
            ]
            for r in d["runs"]
        ]
        _table(
            rows,
            ["ID", "KIND", "STATUS", "MODEL", "LEARNER", "COMPUTE", "SCORE", "COST", "SECONDS"],
        )
        if d["differs"]:
            print(f"\n  differs in: {', '.join(d['differs'])}")

    _emit(args, d, human)


def project_delete(args: argparse.Namespace) -> None:
    slug = _slug(args.project)
    if (
        not args.yes
        and input(f"delete {slug} and its runs? type its name to confirm: ") != slug.split("/")[-1]
    ):
        sys.exit("not deleted")
    call("DELETE", f"/projects/{slug}")
    print(f"✓ deleted {slug}")


# -- runs ----------------------------------------------------------------------------------------


def _watch(run_id: int) -> dict[str, Any]:
    after, last_status = 0, None
    while True:
        r = call("GET", f"/runs/{run_id}?after={after}")
        for e in r.get("events") or []:
            after = max(after, e["seq"])
            p = e["payload"]
            if e["type"] == "episode":
                print(
                    f"  episode {p.get('index')}: score {_score(p.get('score'))}  steps {p.get('steps')}"
                )
            elif e["type"] == "round":
                print(
                    f"  round {p.get('round')}: held-out {_score(p.get('eval_score'))}  {'kept' if p.get('kept') else 'reverted'}"
                )
            elif e["type"] == "metric" and p.get("check"):
                mark = {"pass": "✔", "fail": "✘"}.get(p.get("status"), "–")
                print(f"  {mark} {p.get('tier'):<12} {p.get('check')}")
            elif e["type"] == "progress":
                print(f"  … {p.get('stage')}")
            elif e["type"] == "log":
                print(f"  {p.get('level', 'info')}: {p.get('message')}")
        if r["status"] != last_status:
            last_status = r["status"]
            if r["status"] == "queued":
                print("  waiting for compute…")
        if r["status"] in ("succeeded", "failed", "cancelled"):
            res = r.get("result") or {}
            bits = [
                f"score {_score(res['score'])}" if "score" in res else "",
                f"tier {res['tier']}" if "tier" in res else "",
            ]
            cost = (res.get("compute") or {}).get("cost_usd")
            bits.append(f"${cost:.5f} compute" if cost is not None else "")
            print(
                f"{'✓' if r['status'] == 'succeeded' else '✘'} run {run_id} {r['status']}  "
                + "  ".join(b for b in bits if b)
            )
            if r.get("error"):
                print("  " + r["error"].strip().replace("\n", "\n  "))
            return r
        time.sleep(1.5)


def runs_list(args: argparse.Namespace) -> None:
    runs = (
        call("GET", f"/projects/{_slug(args.project)}/runs")
        if args.project
        else call("GET", "/runs")
    )
    if args.status:
        runs = [r for r in runs if r["status"] == args.status]
    _emit(
        args,
        runs,
        lambda d: _table(
            [
                [
                    r["id"],
                    r["kind"],
                    r["status"],
                    (r.get("listing") or {}).get("slug", ""),
                    _score((r.get("result") or {}).get("score")),
                    r["created_at"][:19],
                ]
                for r in d[: args.limit]
            ],
            ["ID", "KIND", "STATUS", "ENVIRONMENT", "SCORE", "CREATED"],
        ),
    )


def runs_view(args: argparse.Namespace) -> None:
    r = call("GET", f"/runs/{args.id}")

    def human(r):
        res = r.get("result") or {}
        no_model = r["kind"] == "baseline" and not r["config"].get("model")
        kind = "answer-key check" if no_model else r["kind"]
        print(f"run {r['id']}  {kind}  {r['status']}")
        if prov := res.get("provenance"):
            keys = ("compute", "instance", "gpu", "region")
            print("  ran on       " + " · ".join(str(prov[k]) for k in keys if prov.get(k)))
        print(f"  environment  {(r.get('listing') or {}).get('slug', '')}")
        for k in ("score", "baseline", "best", "tier", "episodes"):
            if k in res:
                print(f"  {k:<12} {res[k]}")
        if res.get("compute"):
            c = res["compute"]
            print(f"  compute      {c['seconds']}s · {c['vcpu_hours']} vCPU-h · ${c['cost_usd']}")
        if r.get("error"):
            print("  error        " + r["error"].strip().replace("\n", "\n               "))
        print(f"  events       {len(r.get('events') or [])} (nenyax runs watch {r['id']})")

    _emit(args, r, human)


def runs_watch(args: argparse.Namespace) -> None:
    _watch(args.id)


def _size(n: Any) -> str:
    if not isinstance(n, int):
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return ""


def runs_artifacts(args: argparse.Namespace) -> None:
    rows = call("GET", f"/runs/{args.id}/artifacts")
    _emit(
        args,
        rows,
        lambda d: _table(
            [
                [a["id"], a["kind"], a["name"], _size(a.get("bytes")), a.get("stored_in", "")]
                for a in d
            ],
            ["ID", "KIND", "NAME", "SIZE", "STORED IN"],
        ),
    )


def artifacts_download(args: argparse.Namespace) -> None:
    import urllib.error
    import urllib.request

    url, token = pf.credentials()
    meta = call("GET", f"/artifacts/{args.id}")
    target = args.output or meta["name"]

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **kw):  # handle redirects ourselves, below
            return None

    req = urllib.request.Request(
        f"{url}/artifacts/{args.id}/download", headers={"Authorization": f"Bearer {token}"}
    )
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        resp = opener.open(req, timeout=600)
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303, 307, 308):
            raise
        # A presigned storage URL: fetch it WITHOUT our token (S3 rejects it; it must not leak).
        resp = urllib.request.urlopen(e.headers["Location"], timeout=600)
    with resp:
        if "json" in (resp.headers.get("Content-Type") or ""):
            info = json.loads(resp.read())
            print(f"{meta['name']} lives at {info.get('uri')}; {info.get('note', '')}")
            return
        with open(target, "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
    print(f"✓ saved {target} ({_size(meta.get('bytes'))})")


def runs_cancel(args: argparse.Namespace) -> None:
    r = call("POST", f"/runs/{args.id}/cancel")
    print(f"run {r['id']} {r['status']}")


# -- connections, runners, orgs ------------------------------------------------------------------


def conn_services(args: argparse.Namespace) -> None:
    data = call("GET", "/services", auth=False)
    _emit(
        args,
        data,
        lambda d: _table(
            [[s["id"], s["category"], s.get("env") or "", s["name"]] for s in d],
            ["SERVICE", "KIND", "ENV VAR", "NAME"],
        ),
    )


def conn_list(args: argparse.Namespace) -> None:
    data = call("GET", "/connections")
    _emit(
        args,
        data,
        lambda d: _table(
            [
                [
                    c["id"],
                    c["service"],
                    c["label"],
                    f"…{c['hint']}" if c.get("hint") else "",
                    c["status"],
                    len((c.get("config") or {}).get("models") or []),
                ]
                for c in d
            ],
            ["ID", "SERVICE", "LABEL", "KEY", "STATUS", "MODELS"],
        ),
    )


def conn_add(args: argparse.Namespace) -> None:
    secret = args.key
    if secret == "-":
        secret = sys.stdin.readline().strip()
    elif secret is None and not args.base_url:
        secret = getpass.getpass(f"{args.service} API key (input hidden): ").strip() or None
    body = {
        "service": args.service,
        "label": args.label,
        "secret": secret,
        "base_url": args.base_url,
        "org": args.org,
    }
    c = call("POST", "/connections", {k: v for k, v in body.items() if v is not None})
    c = call("POST", f"/connections/{c['id']}/verify")
    models = len((c.get("config") or {}).get("models") or [])
    mark = "✓" if c["status"] == "ok" else "✘"
    print(
        f"{mark} {c['service']} connection {c['id']}: {c['status']}"
        + (f", {models} models" if models else "")
    )
    if c.get("status_detail") and c["status"] != "ok":
        print(f"  {c['status_detail']}")


def conn_verify(args: argparse.Namespace) -> None:
    c = call("POST", f"/connections/{args.id}/verify")
    print(f"{c['service']} {c['id']}: {c['status']}  {c.get('status_detail') or ''}")


def conn_models(args: argparse.Namespace) -> None:
    for m in call("GET", f"/connections/{args.id}/models"):
        print(m)


def conn_remove(args: argparse.Namespace) -> None:
    call("DELETE", f"/connections/{args.id}")
    print(f"✓ removed connection {args.id}")


def runners_list(args: argparse.Namespace) -> None:
    data = call("GET", "/runners")
    _emit(
        args,
        data,
        lambda d: _table(
            [
                [
                    r["id"],
                    r["name"],
                    "online" if r["online"] else "offline",
                    (r.get("info") or {}).get("gpu") or "",
                    (r.get("last_seen") or "")[:19],
                ]
                for r in d
            ],
            ["ID", "NAME", "STATUS", "GPU", "LAST SEEN"],
        ),
    )


def runners_add(args: argparse.Namespace) -> None:
    r = call("POST", "/runners", {"name": args.name})
    print(
        f"✓ runner {r['id']} ({r['name']}). Start it on the machine that should run jobs:\n\n  {r['command']}\n"
    )


def runners_remove(args: argparse.Namespace) -> None:
    call("DELETE", f"/runners/{args.id}")
    print(f"✓ revoked runner {args.id}")


def org_list(args: argparse.Namespace) -> None:
    data = call("GET", "/me/orgs")
    _emit(
        args,
        data,
        lambda d: _table(
            [[o["org"]["handle"], o["org"]["name"], o["role"]] for o in d], ["ORG", "NAME", "ROLE"]
        ),
    )


def org_create(args: argparse.Namespace) -> None:
    o = call("POST", "/orgs", {"handle": args.handle, "name": args.name or args.handle})
    print(f"✓ created @{o['handle']}")


def org_add(args: argparse.Namespace) -> None:
    call("POST", f"/orgs/{args.org}/members", {"user": args.user, "role": args.role})
    print(f"✓ {args.user} is now {args.role} of @{args.org}")


def api_cmd(args: argparse.Namespace) -> None:
    body = None
    if args.input:
        if args.input == "-":
            body = json.load(sys.stdin)
        else:
            with open(args.input) as f:
                body = json.load(f)
    elif args.field:
        body = _params(args.field)
    out = call(
        args.method.upper(), args.path if args.path.startswith("/") else "/" + args.path, body
    )
    print(json.dumps(out, indent=2, default=str))


# -- wiring --------------------------------------------------------------------------------------


def register(sub: argparse._SubParsersAction) -> None:
    """Add the platform command groups to the ``nenyax`` CLI."""

    def group(name: str, help: str) -> argparse._SubParsersAction:
        p = sub.add_parser(name, help=help)
        g = p.add_subparsers(dest=f"{name}_command", required=True, metavar="command")
        return g

    def cmd(g, name, fn, help, *aliases) -> argparse.ArgumentParser:
        p = g.add_parser(name, help=help, aliases=list(aliases))
        p.set_defaults(func=fn)
        p.add_argument("--json", action="store_true", help="print JSON")
        return p

    g = group("auth", "sign in to the platform (browser, or a token for CI)")
    p = cmd(g, "login", auth_login, "sign in; opens the browser to approve this machine")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--with-token", metavar="TOKEN", help="use a token instead (- reads stdin)")
    p.add_argument("--name", default="nenyax CLI", help="what this token is called in Settings")
    p.add_argument("--no-browser", action="store_true", help="print the link instead of opening it")
    cmd(g, "status", auth_status, "who you are signed in as")
    cmd(g, "logout", auth_logout, "forget the stored token")
    cmd(g, "token", auth_token, "print the token (for scripts)")

    g = group("env", "environments: find, view, create, push, star, try, check")
    p = cmd(g, "list", env_list, "list or search the catalog", "search")
    p.add_argument("query", nargs="?")
    p.add_argument("--kind")
    p.add_argument("--source")
    p.add_argument("--mine", action="store_true", help="only yours, private included")
    p.add_argument("--limit", type=int, default=20)
    p = cmd(g, "view", env_view, "show an environment")
    p.add_argument("slug")
    p.add_argument("--readme", action="store_true")
    p.add_argument("--web", action="store_true", help="open it in the browser")
    p = cmd(g, "new", env_new, "create an environment folder")
    p.add_argument("name")
    p.add_argument("--dest", default=".")
    p = cmd(g, "push", env_push, "publish a folder as a new version")
    p.add_argument("path", nargs="?", default=".")
    p.add_argument("--public", action="store_true")
    p.add_argument("--org")
    p = cmd(g, "star", env_star, "star (or --undo) an environment")
    p.add_argument("slug")
    p.add_argument("--undo", action="store_true")
    for kind, help in (
        ("try", "run one episode on Nenyax Cloud"),
        ("check", "run the conformance suite"),
        ("baseline", "measure a model"),
    ):
        p = cmd(g, kind, lambda a, k=kind: env_run(a, k), help)
        p.add_argument("slug")
        p.add_argument("--model", help="provider/model, e.g. openai/gpt-4o-mini")
        p.add_argument("--param", "-p", action="append", metavar="KEY=VALUE")
        p.add_argument("--watch", "-w", action="store_true")

    g = group("project", "projects: create, view, edit, run, export, delete")
    cmd(g, "list", project_list, "your projects", "ls")
    p = cmd(g, "create", project_create, "create a project")
    p.add_argument("name")
    p.add_argument("--env", help="a catalog slug (owner/name) or your own URI (driver:target)")
    p.add_argument("--model", help="provider/model, e.g. openai/gpt-4o-mini")
    p.add_argument("--learner", help="incontext, grpo, dapo, ... or pkg.module:Learner")
    p.add_argument("--install", help="pip or git spec for a custom learner")
    p.add_argument("--goal", default="llm", choices=["llm", "agent", "control", "robot"])
    p.add_argument("--org")
    p.add_argument("--public", action="store_true")
    p.add_argument(
        "--runners", action="store_true", help="run on your own runners, not Nenyax Cloud"
    )
    p = cmd(g, "view", project_view, "show a project and its next step")
    p.add_argument("project")
    p.add_argument("--web", action="store_true")
    p = cmd(
        g,
        "edit",
        project_edit,
        "change a project's environment, model, learner, pipeline or compute",
    )
    p.add_argument("project")
    p.add_argument("--env")
    p.add_argument("--model")
    p.add_argument("--learner")
    p.add_argument("--install")
    p.add_argument("--pipeline", help="comma-separated steps, e.g. baseline,train,check")
    p.add_argument("--compute", choices=["cloud", "runners"])
    p.add_argument("--visibility", choices=["private", "public"])
    p.add_argument("--description")
    p = cmd(g, "run", project_run, "run the pipeline, or one step")
    p.add_argument("project")
    p.add_argument("step", nargs="?", choices=["try", "baseline", "train", "check"])
    p.add_argument("--param", "-p", action="append", metavar="KEY=VALUE")
    p.add_argument("--watch", "-w", action="store_true", help="stream progress until it finishes")
    p = cmd(g, "export", project_export, "the project as nenyax.toml, to run locally")
    p.add_argument("project")
    p.add_argument("--output", "-o")
    p = cmd(g, "stack", project_stack, "show or change which plug fills each slot")
    p.add_argument("project")
    p.add_argument(
        "--compute",
        help="nenyax-cloud, runners, docker, kubernetes, ssh, modal, runpod, lambda, aws, gcp, prime-compute",
    )
    p.add_argument("--compute-connection", type=int, metavar="ID")
    p.add_argument("--gpu", help="e.g. A100, H100, L4")
    p.add_argument("--gpu-count", type=int)
    p.add_argument("--cpu", type=float)
    p.add_argument("--memory", type=int, metavar="MB")
    p.add_argument("--region")
    p.add_argument("--compute-config", action="append", metavar="KEY=VALUE")
    p.add_argument("--sandbox", help="nenyax, process, docker, e2b, daytona, modal, runloop")
    p.add_argument("--sandbox-connection", type=int, metavar="ID")
    p.add_argument("--trainer", help="nenyax, trl, verl, prime-rl, openrlhf, unsloth, tinker")
    p.add_argument("--trainer-connection", type=int, metavar="ID")
    p.add_argument("--trainer-config", action="append", metavar="KEY=VALUE")
    p.add_argument("--serving", help="api, vllm, sglang")
    p.add_argument("--storage", help="nenyax, s3, gcs, r2, hf")
    p.add_argument("--storage-connection", type=int, metavar="ID")
    p.add_argument(
        "--tracking", action="append", metavar="PLUG:CONNECTION", help="e.g. wandb:12 (repeatable)"
    )
    p.add_argument("--fallback", help="compute plugs to try next, e.g. runpod,modal")
    p.add_argument("--prefer", choices=["first", "cheapest", "fastest"])
    p.add_argument("--max-cost", type=float, metavar="USD", help="per-run budget")
    p.add_argument("--budget", type=float, metavar="USD", help="per-project budget")
    p = cmd(g, "delete", project_delete, "delete a project")
    p.add_argument("project")
    p.add_argument("--yes", action="store_true")

    g = group("runs", "runs: list, view, watch, cancel")
    p = cmd(g, "list", runs_list, "recent runs", "ls")
    p.add_argument("--project")
    p.add_argument("--status")
    p.add_argument("--limit", type=int, default=20)
    for name, fn, help in (
        ("view", runs_view, "show a run"),
        ("watch", runs_watch, "stream a run until it ends"),
        ("cancel", runs_cancel, "cancel a run"),
        ("artifacts", runs_artifacts, "what a run produced (datasets, prompts, checkpoints)"),
    ):
        p = cmd(g, name, fn, help)
        p.add_argument("id", type=int)

    p = cmd(g, "compare", runs_compare, "runs side by side: scores, where they ran, cost")
    p.add_argument("ids", type=int, nargs="+")

    g = group("compute", "where runs execute: plugs and your compute accounts")
    cmd(g, "list", compute_list, "compute plugs and your accounts", "ls")
    p = cmd(g, "test", compute_test, "check a compute connection (no machine is started)")
    p.add_argument("id", type=int)

    g = group("artifacts", "files runs produced: download them")
    p = cmd(g, "download", artifacts_download, "download an artifact")
    p.add_argument("id", type=int)
    p.add_argument("--output", "-o", help="where to save it (default: its name)")

    g = group("connection", "model provider keys, hub accounts and telemetry")
    cmd(g, "services", conn_services, "everything you can connect")
    cmd(g, "list", conn_list, "your connections", "ls")
    p = cmd(g, "add", conn_add, "connect a service (the key is prompted, never echoed)")
    p.add_argument("service", help="openai, anthropic, together, huggingface, vllm, wandb, ...")
    p.add_argument("--key", help="the secret (- reads stdin); prompted if omitted")
    p.add_argument("--base-url", help="for local and self-hosted servers")
    p.add_argument("--label")
    p.add_argument("--org", help="share with an organization")
    for name, fn, help in (
        ("verify", conn_verify, "check it works"),
        ("models", conn_models, "its models"),
        ("remove", conn_remove, "delete it"),
    ):
        p = cmd(g, name, fn, help)
        p.add_argument("id", type=int)

    g = group("runners", "your own compute (Nenyax Cloud needs none)")
    cmd(g, "list", runners_list, "your runners", "ls")
    p = cmd(g, "add", runners_add, "create a runner token and print its start command")
    p.add_argument("name")
    p = cmd(g, "remove", runners_remove, "revoke a runner")
    p.add_argument("id", type=int)

    g = group("org", "organizations")
    cmd(g, "list", org_list, "organizations you belong to", "ls")
    p = cmd(g, "create", org_create, "create an organization")
    p.add_argument("handle")
    p.add_argument("--name")
    p = cmd(g, "add-member", org_add, "add someone (handle or email)")
    p.add_argument("org")
    p.add_argument("user")
    p.add_argument("--role", default="member", choices=["member", "admin", "owner"])

    p = sub.add_parser("api", help="call any platform endpoint, like `gh api`")
    p.add_argument(
        "method",
        choices=["GET", "POST", "PATCH", "PUT", "DELETE", "get", "post", "patch", "put", "delete"],
    )
    p.add_argument("path", help="e.g. /projects or /listings?q=sudoku")
    p.add_argument("--field", "-f", action="append", metavar="KEY=VALUE", help="JSON body fields")
    p.add_argument("--input", help="JSON body file (- for stdin)")
    p.set_defaults(func=api_cmd)
