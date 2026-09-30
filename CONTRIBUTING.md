# Contributing to Nenyax

Nenyax is neutral infrastructure: every environment format, sandbox, model provider, trainer,
judge and telemetry tool should be able to plug in, whether it is open source or a commercial
service. There are two ways to bring yours in.

## Route A: your own package (recommended for companies and vendors)

You keep ownership, release on your own schedule, and need no change to Nenyax.

```bash
nenyax new-plugin sandbox acme        # or: driver | learner | judge | hub | telemetry
cd nenyax-acme && pip install -e '.[dev]' && pytest     # passes on day one
```

1. Replace the stub with your integration. Import your SDK lazily, and take credentials from
   environment variables (declare them in `credentials = (...)`).
2. Keep `tests/test_conformance.py` green. It runs the same `nenyax.testing` check that built-in
   plugins must pass.
3. Publish to PyPI. Once installed, users see it in `nenyax plugins`, and can pick it by name
   in Python or config (`use = "acme"`).
4. **Get listed:** open a PR here that adds one `[[plugin]]` entry to
   [`registry/plugins.toml`](registry/plugins.toml). Run
   `python tools/validate_registry.py --install` first: it installs your package and checks that its
   entry point loads, and maintainers run the same check before merging.

Extension points:

| Kind | Entry-point group | You implement | Conformance |
|---|---|---|---|
| driver | `nenyax.drivers` | `Driver.load() -> Environment` | `assert_conforms(env, tier)` |
| sandbox | `nenyax.sandboxes` | `SandboxBackend.create() -> Sandbox` | `assert_sandbox_conforms` |
| provider | `nenyax.providers` | `Provider` (an OpenAI-compatible endpoint) | `assert_provider_conforms` |
| hub | `nenyax.hubs` | `Hub.search()` / `fetch()` | `assert_hub_conforms` |
| learner | `nenyax.learners` | `policy()` + `update(groups)` | `assert_learner_conforms` |
| judge | `nenyax.judges` | `judge(traj) -> Judgment` | `assert_judge_conforms` |
| telemetry | `nenyax.telemetry` | `on_trajectory(traj)` | `assert_callback_conforms` |

`nenyax plugins --kinds` prints this table from the installed version.

## Route B: a PR into Nenyax core

For integrations that are open source and widely used, or for changes to the contract itself.

```bash
uv venv && uv pip install -e '.[dev,gymnasium,reasoning-gym,openenv,verifiers,harbor]'
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```

A core plugin PR includes:

1. The implementation, in the directory for its kind (`src/nenyax/formats/`,
   `src/nenyax/sandbox/`, `src/nenyax/integrations/`, `src/nenyax/learners/`).
2. An integration test that exercises the **real** package or service and asserts the tier or
   conformance it actually reaches. Keyed services skip when the key is absent and are marked
   untested-live until someone runs them.
3. A row in the README's tables, stating the status you measured.

Changes to `nenyax.types` or other contract files need a maintainer review (see
`.github/CODEOWNERS`) and must stay backward compatible, or be called out in the PR.

## Principles

- **Translate, don't replace.** Never ask environment authors to change their native format.
- **Never over-claim.** Capabilities, tiers and "verified" labels reflect what tests measured.
- **Bring your own key.** Nenyax never stores credentials and never phones home.
- **Failures are data.** Rollout errors belong in `Trajectory.error`, not in exceptions.
- **Lean core.** Vendor-specific code lives in plugins, and optional dependencies stay optional.
