# What we learned building Nenyax v0.1

These findings come from building drivers for six formats and running them against real packages:
Gymnasium 1.3, reasoning-gym 0.1.25, OpenEnv 0.6, NeMo Gym 0.6, Verifiers 0.3.1 and Harbor 0.23.
We also ran one real model, Qwen2.5-1.5B-Instruct served by mlx-lm, across all of them. Dates: September 2026.

## Results by format

| Format | Mode | Tier reached | What made it work | What blocked or surprised us |
|---|---|---|---|---|
| reasoning-gym | step | audited | Native `score_answer` is a real verifier, and the gold answer gives an oracle for free | Answer formats vary; we standardized on `<answer>` tags |
| Gymnasium | step | trainable | `reset(seed)`/`step` maps one to one | No oracle exists, so the verifier cannot be audited, and that is correct |
| OpenEnv | step | evaluable (public Spaces), trainable (seeded servers) | One `GenericEnvClient` plus `/schema` drove every server, with no per-env code | Servers don't declare whether `reset(seed)` is honored; public Spaces cap sessions, sleep or break; Wordle's `state()` leaks the answer |
| NeMo Gym | step | audited | The resources server's `/verify` is a clean, stable contract over HTTP | Needs Python ≥ 3.13.14, Ray and a venv per server (2.8 GB for 3 servers); talking HTTP only kept all of that out of our process |
| Verifiers | rollout | audited | Both stacks work: legacy `run_rollout` and v1 `run_slot`; the Environments Hub `gsm8k` (1,319 tasks) scored gold 1 / wrong 0 | Two incompatible stacks in one package; v1 defaults need Prime Intellect auth and remote sandboxes; tool calls arrive in OpenAI's nested wire shape |
| Harbor | task | audited | `oracle`/`nop` give a built-in verifier audit; `terminus-2` accepts any OpenAI-compatible `api_base` | About 4–25 s per episode (container plus agent setup); rewards only after the episode ends, so it cannot be branched |

## Findings

| # | Finding | Evidence | What we did |
|---|---|---|---|
| 1 | **Every format fits one of three modes**: step, rollout or task | 6 formats, 0 exceptions | Made `mode` a required manifest field |
| 2 | **One universal call is enough**: `rollout(policy)` | The same `Endpoint` object drove 8 environments across 6 formats | `session()` exists only for step mode; the other modes raise `UnsupportedCapability` |
| 3 | **Recording is only possible because Nenyax owns the model endpoint.** Environments that run their own loop are otherwise opaque to trainers | 100% of model calls recorded across modes after the fix below | A per-episode recording server at `/s/<session>/v1` |
| 4 | Our first cut forgot step mode: chat models driving step environments bypassed recording | The real-model run showed 0 calls on step rows | `RecordingModel` wraps every chat model; tokens and logprobs are kept |
| 5 | **Process boundaries beat shared dependencies.** NeMo Gym (Python 3.13, Ray) could not share a venv, but its HTTP contract could | The NeMo Gym driver has zero Python dependencies | Prefer HTTP drivers when the native stack is heavy |
| 6 | **Plug-and-play is a spectrum, so measure it.** The same driver yields different tiers per environment | Public OpenEnv Spaces are evaluable, seeded servers trainable, reasoning-gym audited | Conformance tiers with skip-blocks-tier semantics |
| 7 | **Reproducibility is rarely declared.** Nothing in OpenEnv's `/schema` says whether seeds are honored | Checked every OpenEnv server we tried | `resettable` is false unless the user asserts `seeded=True` |
| 8 | **Verifier audits need an oracle, and most formats have one hidden somewhere**: gold answers, `solution/` directories, built-in agents | reasoning-gym, NeMo Gym, Verifiers and Harbor all reached audited | `reference_policy`/`null_policy` hooks; Harbor uses `oracle`/`nop` |
| 9 | **The audit must be task-aware.** Oracles know one task's answer, not all of them | The first audit was wrongly skipped for Verifiers | `reference_policy(task)` is called per task |
| 10 | **The model-path check must use a model.** An oracle agent never calls one | Harbor was first mis-graded as "bypassed the endpoint" | Separate `probe_policy()` that speaks the agent's reply format |
| 11 | **OpenAI's chat wire format is the real lingua franca**, including its nested tool-call shape | Verifiers `ToolEnv` failed until we parsed `{"type":"function","function":{...}}` | `Message.to_openai()` and a validator that accepts both shapes |
| 12 | **Port-out works**: any step environment can be served as OpenEnv | reasoning-gym → OpenEnv server → back in, score and judgment preserved | `nenyax serve` |
| 13 | **Low scores from a small real model were genuine, not bugs.** Qwen2.5-1.5B answered "2+2 equals 4." with no `<answer>` tags | Transcripts inspected | Format-following is exactly what RL on these environments teaches; honest verifiers matter |
| 14 | The frameworks coexist in one venv (except NeMo Gym) | verifiers, harbor, openenv, gymnasium and reasoning-gym installed together | Optional extras per driver |

## Sandboxes

| # | Finding | What we did |
|---|---|---|
| 15 | The model, the agent loop and the environment are separate; only environments with dangerous actions need a sandbox | `Capabilities.isolation`; `select("none")` returns no sandbox |
| 16 | Every provider (and Harbor's and NeMo Gym's own interfaces) converges on create / exec / files / destroy | That is the whole required contract; fork and ports are optional and declared |
| 17 | Forking mid-episode is the RL-relevant differentiator, and no RL framework exposes it | `BackendInfo.fork` = none / disk / memory; conformance checks that a child cannot leak into its parent |
| 18 | Docker cannot publish ports from a no-network container | The backend refuses the combination instead of silently opening the network |
| 19 | Claims must be tested: "network off" is only true if an outbound connection actually fails, and the probe tool must exist | The network check first verifies it has a probe tool |

### Adversarial and load campaign (both backends, Apple M3 Pro)

| # | Finding | Fix |
|---|---|---|
| 20 | A 300 MB output flood was buffered whole in the controller (+1.3 GB RSS), so one chatty agent could OOM a trainer | Output is capped per stream (`SandboxSpec.output_cap`, 1 MiB default), pipes keep draining, and `ExecResult.truncated` is set. RSS stays flat at 38 MB |
| 21 | An OOM kill was reported as a timeout | Timeouts exit 124, other SIGKILLs are reported as `killed` |
| 22 | Relative paths broke the Docker backend (`write_file("x")` made a directory); conformance only tested absolute paths | Paths resolve against the workdir; conformance now checks relative paths and a 1 MiB binary round trip |
| 23 | Background children holding the pipes lost short output (`read(n)` waited for n bytes) | `read1` plus one shared 2 s grace period |
| 24 | A fork bomb is contained by Docker (`--pids-limit`) but leaves that sandbox unusable | Treat sandboxes as disposable; pools hand out fresh ones |
| 25 | The `process` backend does not limit process count, memory (macOS) or disk | Declared honestly; use it only for trusted-ish actions. Attacks it does not claim to stop were not run against the host |
| 26 | Docker enforces no disk quota on this setup | Not claimed in `enforces` |

| Load test | Result |
|---|---|
| `process`, 200 episodes, 32 parallel | 167 episodes/s, p95 0.31 s, zero cross-talk |
| `docker`, 40 episodes, 8 parallel | 1.3 episodes/s, p50 6.5 s (create/destroy on Docker Desktop dominates) |
| `docker` warm pool | fill 12 s once (disk forks via `commit` about 2 s each), then 0.1 ms per acquire |
| Crash while holding 3 containers | `cleanup()` reclaimed all 3 |
| Real model (Qwen2.5-1.5B) writes code, hidden tests grade it in the sandbox | 8/8 on `process`, 8/8 on `docker` |

### Integrations

| # | Finding | Consequence |
|---|---|---|
| 27 | All three environment hubs are publicly searchable: HF Spaces (`openenv` tag), the Prime Environments Hub (1,775 environments) and the Harbor registry (80 datasets) | Search and fetch need no key; Terminal-Bench 2 tasks were fetched and loaded |
| 28 | HF Space hostnames can't be derived from the name (collisions get a suffix), and most OpenEnv Spaces are asleep | Use the API's `subdomain` and report `runtime.stage` |
| 29 | `nenyax.drivers()` was shadowed by the `nenyax.drivers` subpackage | Subpackage renamed to `nenyax.formats` |
| 30 | Training platforms vary most. OpenAI RFT is closed to new users; Prime's shared hosted RL stops new runs 2026-10-05; Applied Compute and Together have no public RL API | "We run rollouts, platform trains" (Tinker, Fireworks Training API, CoreWeave ART, Nebius, TRL) is the widest integration shape |
| 31 | MLflow's file store is in maintenance mode | Use `sqlite:///` tracking URIs |

### Learning

| # | Finding | Consequence |
|---|---|---|
| 32 | The in-context learner lifted a real Qwen2.5-1.5B on held-out tagged Q&A from 0.50 to 1.00 in two rounds, with no weight updates | "Improve" works for closed API models too |
| 33 | `train()` once reported a clean 0.000 while every episode was erroring (wrong port) | It now raises `TrainingError` with the first error, and counts errors per round |
| 34 | GRPO on a confident 0.5B model: most groups had identical rollouts, so there was no signal | The `signal` metric exposes it. Harder tasks or bigger groups are needed |
| 35 | The checkpoint's generation defaults (top_k=20, top_p=0.8) were silently applied during RL sampling, which makes the loss off-policy and suppresses exploration | GRPO samples from the full distribution |
| 36 | The same shadowing bug hit twice (`nenyax.drivers`, `integrations.hubs`): package attributes vs. functions with the same name | Avoid function names that equal submodule names |

## Timings (Apple M3 Pro, local)

| Environment | Per episode |
|---|---|
| reasoning-gym (scripted policy) | < 1 ms |
| NeMo Gym resources server over HTTP | ~20 ms |
| OpenEnv public HF Space (9-step game) | ~0.3 s per step |
| Verifiers legacy / v1 (local model) | ~0.5 s / ~1–4 s (v1 starts its interception proxy per episode) |
| Harbor oracle / terminus-2 | ~4 s / ~16–25 s |

## What to improve next

| Priority | Improvement | Why |
|---|---|---|
| 1 | NeMo Gym agent servers (multi-turn, tools) | Most of NeMo Gym's 176 servers are beyond single-turn `/verify` |
| 2 | Pool Verifiers v1 `serving()` and Harbor containers across episodes | Per-episode setup dominates wall-clock time |
| 3 | Propose `Capabilities` (especially `resettable`) and tiers to OpenEnv as an RFC | Finding 7: the ecosystem has no way to declare seed support |
| 4 | Branchable sessions (snapshot and fork) for step environments that support it | Needed for tree search and per-step credit assignment |
| 5 | A bulk `nenyax audit` over whole hubs (Prime Intellect, OpenEnv Spaces, Harbor registry) | Publish the real distribution of tiers across public environments |
| 6 | More drivers: PettingZoo/OpenSpiel (multi-agent), BrowserGym, ORS/OpenReward, dm_env | Long tail |
| 7 | Trainer bindings: TRL `environment_factory`, verl agent loop, prime-rl | Close the loop from trajectory to gradient |
| 8 | Contamination and leakage checks in the audit tier | The Wordle `state()` leak and SWE-bench future-commit leaks are real |
| 9 | Docker throughput: reuse and reset containers instead of create/destroy; faster forks than `commit` | 1.3 episodes/s under load on Docker Desktop |
| 10 | Linux isolation backends (bubblewrap/nsjail jail, gVisor, Firecracker memory fork) | The `process` backend's gaps (process count, memory, disk) need a jail on Linux |
