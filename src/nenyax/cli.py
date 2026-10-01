"""``nenyax`` command line: inspect, check and run any environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.util import find_spec

from . import __version__
from .conformance import check
from .environment import Environment, Policy
from .policy import Endpoint
from .registry import DriverUnavailable, drivers, load
from .runner import run
from .sandbox import BackendNotConfigured


def _policy(env: Environment, args: argparse.Namespace) -> Policy:
    if args.policy == "endpoint":
        if not (args.base_url and args.model):
            sys.exit("--policy endpoint needs --base-url and --model")
        return Endpoint(args.base_url, args.model, api_key=args.api_key)
    if args.policy == "reference":
        p = env.reference_policy()
    elif args.policy == "null":
        p = env.null_policy()
    else:
        maker = getattr(env, "random_policy", None)
        p = maker(seed=args.seed) if maker else None
    if p is None:
        sys.exit(f"{env.id} provides no '{args.policy}' policy")
    return p


def _cmd_drivers(args: argparse.Namespace) -> None:
    for name, d in sorted(drivers().items()):
        state = "installed" if d.available() else f"pip install 'nenyax[{d.extra}]'"
        print(f"{name:<16} {d.format:<40} {state}")


def _cmd_info(args: argparse.Namespace) -> None:
    with load(args.uri) as env:
        print(env.manifest.model_dump_json(indent=2, exclude_none=True))


def _cmd_check(args: argparse.Namespace) -> None:
    with load(args.uri) as env:
        report = check(env, samples=args.samples)
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.render())


def _cmd_run(args: argparse.Namespace) -> None:
    with load(args.uri) as env:
        policy = _policy(env, args)
        scores = []
        for traj in run(
            env,
            policy,
            n=args.n,
            seed=args.seed,
            out=args.out,
            concurrency=args.concurrency,
            max_steps=args.max_steps,
        ):
            scores.append(traj.score)
            status = "error: " + traj.error if traj.error else f"score={traj.score}"
            task = traj.task.id if traj.task else f"seed {traj.seed}"
            print(
                f"{task:<12} {status}  ({traj.duration_s:.2f}s, "
                f"{len(traj.steps)} steps, {len(traj.model_calls)} model calls)"
            )
    ok = [s for s in scores if s is not None]
    if ok:
        print(f"mean score {sum(ok) / len(ok):.3f} over {len(ok)}/{len(scores)} episodes")


def _cmd_serve(args: argparse.Namespace) -> None:
    from .export.openenv import serve

    print(f"serving {args.uri} as OpenEnv on http://{args.host}:{args.port}")
    serve(args.uri, host=args.host, port=args.port)


def _cmd_sandboxes(args: argparse.Namespace) -> None:
    from . import sandbox

    for name, cls in sorted(sandbox.backends().items()):
        info, missing = cls.info, cls().missing()
        state = "ready" if not missing else "; ".join(missing)
        print(f"{name:<14} {info.isolation:<10} fork={info.fork:<7} {state}")


def _cmd_sandbox_check(args: argparse.Namespace) -> None:
    from . import sandbox

    report = sandbox.check_backend(sandbox.get(args.backend), sandbox.SandboxSpec(image=args.image))
    print(report.render())
    sys.exit(0 if report.ok else 1)


def _cmd_integrations(args: argparse.Namespace) -> None:
    from . import integrations as I
    from . import sandbox
    from .integrations import telemetry as T

    rows = [(p.kind, n, p.missing()) for n, p in I.providers().items()]
    rows += [(h.kind, n, h.missing()) for n, h in I.hubs().items()]
    rows += [
        ("telemetry", c.name, [f"pip install {m}" for m in c.modules if find_spec(m) is None])
        for c in (T.OpenTelemetry, T.MLflow, T.WandB)
    ]
    rows += [("sandbox", n, c().missing()) for n, c in sandbox.backends().items()]
    learner_needs = {
        "incontext": [],
        **{
            n: [f"pip install {m}" for m in ("torch", "transformers") if find_spec(m) is None]
            for n in (
                "grpo",
                "dr_grpo",
                "dapo",
                "rloo",
                "reinforce",
                "reinforce_pp",
                "gspo",
                "pg",
                "rft",
                "dpo",
            )
        },
        "tinker": ([] if find_spec("tinker") else ["pip install tinker"])
        + ([] if os.environ.get("TINKER_API_KEY") else ["set TINKER_API_KEY"]),
    }
    from . import config

    rows += [("learner", n, learner_needs.get(n, [])) for n in config.registered("learner")]
    rows += [("judge", n, []) for n in config.registered("judge")]
    for kind, name, missing in rows:
        print(f"{kind:<10} {name:<14} {'ready' if not missing else '; '.join(missing)}")


def _cmd_search(args: argparse.Namespace) -> None:
    from . import integrations as I

    for hit in I.search(args.hub, args.query, limit=args.limit):
        extra = hit.meta.get("stage") or ""
        print(f"{hit.uri:<72} {extra:<9} {hit.description[:60]}")


def _model_from(args: argparse.Namespace):
    from . import integrations as I

    if not args.model:
        sys.exit("--model is required (and --provider, e.g. vllm, ollama, openrouter)")
    params = {"temperature": args.temperature, "max_tokens": args.max_tokens}
    return I.endpoint(args.provider, args.model, base_url=args.base_url, **params)


def _cmd_try(args: argparse.Namespace) -> None:
    from .replay import render

    with load(args.uri) as env:
        if args.model:
            policy = _model_from(args)
        else:
            policy = env.reference_policy() or sys.exit(
                "no --model given and the environment has no reference policy"
            )
        tasks = env.tasks(limit=args.n) or [None] * args.n
        for i, task in enumerate(tasks[: args.n]):
            print(render(env.rollout(policy, task=task, seed=args.seed + i)))


def _cmd_train(args: argparse.Namespace) -> None:
    from .learners import InContextLearner
    from .train import train

    if args.config:
        from . import config

        print(config.run(args.config).render())
        return
    if not args.uri:
        sys.exit("give an environment URI or --config run.toml")
    with load(args.uri) as env:
        if args.learner == "grpo":
            from .learners import GRPOLearner

            learner = GRPOLearner(
                args.model, lr=args.lr, max_new_tokens=args.max_tokens, temperature=args.temperature
            )
        else:
            learner = InContextLearner(_model_from(args))
        result = train(
            env,
            learner,
            rounds=args.rounds,
            group_size=args.group_size,
            batch=args.batch,
            concurrency=args.concurrency,
            seed=args.seed,
        )
        print(result.render())
        if args.save and hasattr(learner, "save"):
            learner.save(args.save)
            print(f"saved to {args.save}")


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", default="vllm", help="see `nenyax integrations`")
    p.add_argument("--model", help="model name (or a Hugging Face id for --learner grpo)")
    p.add_argument("--base-url", help="override the provider's default URL")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=512)


def _cmd_plugins(args: argparse.Namespace) -> None:
    from .plugins import KINDS, installed

    if args.registry:
        import tomllib
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "registry" / "plugins.toml"
        entries = tomllib.loads(path.read_text()).get("plugin", []) if path.is_file() else []
        for e in entries:
            print(f"{e['kind']:<10} {e['name']:<16} {e['package']:<24} {e['maintainer']}")
        if not entries:
            print("no third-party plugins listed yet: see CONTRIBUTING.md (Route A)")
        return
    if args.kinds:
        for k in KINDS.values():
            print(f"{k.name:<10} {k.group:<18} {k.interface:<34} nenyax.testing.{k.conformance}")
        return
    for p in sorted(installed(), key=lambda p: (p.kind, p.source != "builtin", p.name)):
        print(f"{p.kind:<10} {p.name:<16} {p.source:<16} {p.target}")


def _cmd_new_plugin(args: argparse.Namespace) -> None:
    from .scaffold import scaffold

    path = scaffold(args.kind, args.name, args.dest)
    print(f"created {path}\n  cd {path} && pip install -e . && pytest")


def _cmd_runner(args: argparse.Namespace) -> None:
    import logging

    from .remote import serve

    token = args.token or os.environ.get("NENYAX_RUNNER_TOKEN")
    if not token:
        sys.exit("--token (or NENYAX_RUNNER_TOKEN) is required; create one on the Runners page")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    print(f"runner connecting to {args.url}; jobs run on this machine. Ctrl-C to stop.")
    try:
        serve(args.url, token, once=args.once)
    except KeyboardInterrupt:
        print("runner stopped")


def _cmd_login(args: argparse.Namespace) -> None:
    from . import platform

    platform.save_login(args.url, args.token)
    try:
        me = platform.whoami()
    except platform.PlatformError as e:
        sys.exit(f"saved, but the platform rejected the token: {e}")
    print(f"signed in as {me.get('name')} (@{me.get('handle')}) at {args.url}")


def _cmd_new_env(args: argparse.Namespace) -> None:
    from . import platform

    root = platform.new_env(args.name, args.dest)
    print(f"created {root}/ (env.py, nenyax.toml, README.md)")
    print(f"  try it:   nenyax try {root}")
    print(f"  push it:  nenyax push {root}")


def _cmd_push(args: argparse.Namespace) -> None:
    from . import platform

    try:
        out = platform.push(args.path, public=args.public, org=args.org)
    except platform.PlatformError as e:
        sys.exit(str(e))
    verb = "published" if out["created"] else "updated"
    print(f"{verb} {out['slug']} {out['version']} ({out['bytes'] / 1024:.1f} KB)")
    print(f"  {out['url']}")
    if out.get("check_run"):
        print(f"  conformance check queued on Nenyax Cloud (run {out['check_run']})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nenyax", description=__doc__)
    parser.add_argument("--version", action="version", version=f"nenyax {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("drivers", help="list drivers and whether they are installed")

    p = sub.add_parser("info", help="print an environment's manifest")
    p.add_argument("uri")

    p = sub.add_parser("check", help="run the conformance suite")
    p.add_argument("uri")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("run", help="run episodes with a policy")
    p.add_argument("uri")
    p.add_argument(
        "--policy", choices=["reference", "null", "random", "endpoint"], default="reference"
    )
    p.add_argument("--base-url")
    p.add_argument("--model")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("-n", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--out", help="append trajectories to this JSONL file")

    p = sub.add_parser("serve", help="serve a step-mode environment as an OpenEnv server")
    p.add_argument("uri")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    p = sub.add_parser("try", help="run one or more episodes and show readable replays")
    p.add_argument("uri")
    p.add_argument("-n", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    _add_model_args(p)

    p = sub.add_parser("train", help="improve a model on an environment")
    p.add_argument("uri", nargs="?")
    p.add_argument("--config", help="TOML/YAML/JSON run config (see nenyax.config)")
    p.add_argument(
        "--learner",
        default="incontext",
        help="incontext | grpo | dr_grpo | dapo | rloo | reinforce | reinforce_pp | "
        "gspo | rft | dpo | tinker | module:Class",
    )
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", help="directory to save trained weights (grpo)")
    _add_model_args(p)

    p = sub.add_parser("plugins", help="installed plugins of every kind (built-in and third-party)")
    p.add_argument("--kinds", action="store_true", help="list extension points instead")
    p.add_argument("--registry", action="store_true", help="list the public plugin registry")
    p = sub.add_parser("new-plugin", help="scaffold a plugin package with a conformance test")
    p.add_argument("kind", choices=["driver", "sandbox", "learner", "judge", "hub", "telemetry"])
    p.add_argument("name")
    p.add_argument("--dest", default=".")

    p = sub.add_parser("runner", help="run platform jobs on this machine (outbound only)")
    p.add_argument(
        "--url", default=os.environ.get("NENYAX_PLATFORM_URL", "http://127.0.0.1:4300/api")
    )
    p.add_argument("--token", help="runner token from the Runners page")
    p.add_argument("--once", action="store_true", help="exit after one job (or none queued)")

    p = sub.add_parser("login", help="sign in to a Nenyax platform with a personal access token")
    p.add_argument(
        "--url", default=os.environ.get("NENYAX_PLATFORM_URL", "http://127.0.0.1:4300/api")
    )
    p.add_argument("--token", required=True, help="nxp_... from Settings → Tokens")
    p = sub.add_parser("new-env", help="create an environment folder ready to push")
    p.add_argument("name")
    p.add_argument("--dest", default=".")
    p = sub.add_parser("push", help="publish an environment folder to the platform")
    p.add_argument("path", nargs="?", default=".")
    p.add_argument("--public", action="store_true", help="list it publicly (default: private)")
    p.add_argument("--org", help="publish under an organization you belong to")

    sub.add_parser("integrations", help="every integration and what it still needs")
    p = sub.add_parser("search", help="search an environment hub (huggingface, prime, harbor)")
    p.add_argument("hub")
    p.add_argument("query", nargs="?", default="")
    p.add_argument("--limit", type=int, default=20)
    sub.add_parser("sandboxes", help="list sandbox backends and what they need")
    p = sub.add_parser("sandbox-check", help="verify a sandbox backend delivers its claims")
    p.add_argument("backend")
    p.add_argument("--image", default="python:3.12-slim")

    args = parser.parse_args(argv)
    handler = {
        "drivers": _cmd_drivers,
        "info": _cmd_info,
        "check": _cmd_check,
        "run": _cmd_run,
        "serve": _cmd_serve,
        "sandboxes": _cmd_sandboxes,
        "integrations": _cmd_integrations,
        "runner": _cmd_runner,
        "plugins": _cmd_plugins,
        "new-plugin": _cmd_new_plugin,
        "try": _cmd_try,
        "train": _cmd_train,
        "search": _cmd_search,
        "sandbox-check": _cmd_sandbox_check,
        "login": _cmd_login,
        "new-env": _cmd_new_env,
        "push": _cmd_push,
    }
    try:
        handler[args.command](args)
    except (DriverUnavailable, BackendNotConfigured) as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
