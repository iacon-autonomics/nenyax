# Changelog

All notable changes are listed here. The contract (`nenyax.types`) is versioned separately as
`PROTOCOL_VERSION`.

## 0.1.0a2 (unreleased)

- Package metadata: author, homepage, repository, issues, changelog and Discord links.
- Branded README; security reports go to dev@iaconautonomics.com.
- Docker sandboxes: `advertise_host` for controllers that run inside containers.
- Learners: Dr. GRPO, DAPO, RLOO, REINFORCE, REINFORCE++-style, GSPO, configurable `pg`,
  rejection sampling and DPO.

## 0.1.0a1 (first public alpha, on PyPI)

Contract `0.1`.

### Environments
- One contract (`Manifest`, `Capabilities`, `Trajectory`, `Judgment`) and one call, `rollout()`,
  across three modes: step, rollout and task.
- Drivers for Gymnasium, reasoning-gym, OpenEnv, NeMo Gym resources servers, Verifiers (legacy
  and v1, including Environments Hub packages) and Harbor (Terminal-Bench and adapted benchmarks).
- `nenyax.define()` for environments from a dataset plus a scorer, or from state functions.
- Conformance tiers (connectable, evaluable, trainable, audited), measured by `nenyax check`.
- `nenyax serve`: any step-mode environment as an OpenEnv server.

### Judging and learning
- Judges: env, exact match, regex, Python function, LLM with a rubric, weighted combinations,
  applied to any environment with `with_judge`.
- Learners: in-context (any model, closed APIs included); policy-gradient presets GRPO, Dr. GRPO,
  DAPO, RLOO, REINFORCE, REINFORCE++-style and GSPO, plus a fully configurable `pg`;
  rejection-sampling fine-tuning; DPO; Tinker (hosted, bring your own key).
- `nenyax.train()`: held-out evaluation, a gate that reverts harmful updates, a learning-signal
  metric, and a loud error when every episode fails.

### Infrastructure and integrations
- Sandboxes: `process` and `docker`/`podman` backends, warm pools with forking, and a conformance
  suite that attacks every declared guarantee.
- 18 model providers, 3 environment hubs (Hugging Face OpenEnv Spaces, Prime Environments Hub,
  Harbor registry), OpenTelemetry, MLflow and W&B callbacks, and exports to JSONL, Parquet and
  the HF Hub in raw, SFT or DPO shape.
- Config files: every piece chosen by name or `module:Class`, with `nenyax train --config`.
- Plugins: 7 extension kinds with entry points, `nenyax new-plugin` scaffolds that pass
  conformance on day one, and a public registry with CI validation.

### Verified on
- macOS 26 (Apple M3 Pro) and Linux arm64 (Debian 13), Python 3.12.
- Real packages at the versions pinned in `pyproject.toml`, and real models (Qwen2.5 0.5B and
  1.5B) for the end-to-end runs.

### Not yet verified live (need credentials)
- Tinker learner, hosted model providers, bring-your-own-key sandbox providers.

### Known limitations
- Weight-updating learners are tested for correctness (exact advantage maths, real gradient
  steps) but not yet benchmarked for learning at scale. Treat them as experimental.
- NeMo Gym agent servers (multi-turn and tool rows) are not driven yet; such rows are rejected.
- Linux-only isolation backends (bubblewrap, nsjail, gVisor, Firecracker) are planned.
