# Nenyax

**One contract for every RL environment.** Load an environment from any framework, drive it with
any policy or model, and get back the same judged trajectory, with an honest measurement of how
far that environment actually gets.

```python
import nenyax

model = nenyax.Endpoint("http://localhost:8000/v1", "Qwen/Qwen3-8B")   # any OpenAI-compatible server

for uri in [
    "reasoning_gym:basic_arithmetic?size=100",       # reasoning-gym
    "verifiers:gsm8k",                               # Prime Intellect Environments Hub
    "nemo_gym:http://localhost:8001?data=tasks.jsonl",  # NVIDIA NeMo Gym
    "openenv:https://openenv-openspiel-env.hf.space",   # OpenEnv / Hugging Face Spaces
    "harbor:./terminal-bench/fix-git",               # Harbor / Terminal-Bench
]:
    env = nenyax.load(uri)
    traj = env.rollout(model)                        # same call, same Trajectory, every format
    print(env.manifest.source_format, traj.score, len(traj.model_calls))
```

Nenyax does not replace these frameworks, and it does not define yet another environment format.
It is a thin layer that **routes each native format into one contract**, and can serve step-mode
environments back out over the OpenEnv protocol, so trainers that speak OpenEnv work with all of
them.

## Why

RL environments are scattered across incompatible SDKs: Gymnasium, OpenEnv, Verifiers, NeMo Gym,
Harbor, reasoning-gym and more. Each trainer supports a few of them, each through its own adapter.
Nenyax provides:

1. **One contract.** `Manifest`, `Capabilities`, `Trajectory` and `Judgment` are the same whatever the source format.
2. **One call.** `env.rollout(policy)` works whether the environment is stepped from outside, runs its own agent loop, or is a sandboxed task graded by tests.
3. **Recording by default.** Every model call goes through a recording OpenAI-compatible server, including token ids and logprobs when the upstream returns them. Environments that run their own loop stay usable for training.
4. **Honest conformance.** `nenyax check <uri>` measures what an environment really does, instead of assuming it is plug-and-play.

## Install

```bash
pip install nenyax                       # core: pydantic only
pip install 'nenyax[reasoning-gym]'      # plus any of: gymnasium, verifiers, harbor, openenv
```

NeMo Gym needs no extra: Nenyax talks to its resources servers over HTTP, so NeMo Gym's own Python
3.13 and Ray stay in their own environment. On Linux arm64, `reasoning-gym` builds one dependency
(`pycosat`) from source, so install a C compiler first (`apt install gcc`).

## The two layers

```
  native formats ──► Layer 2: drivers + router ──► Layer 1: the Nenyax contract ──► you / trainers
  (Gymnasium,        one driver per format,         Manifest · Capabilities           rollout()
   OpenEnv,          grouped by how the env         Observation · Step · Judgment     check()
   Verifiers, ...)   is driven                      Trajectory · ModelCall            serve()
```

Every format maps to exactly one **mode**:

| Mode | Who drives the loop | Formats | Stepping from outside |
|---|---|---|---|
| `step` | the caller: `session()` → `act()` | Gymnasium, OpenEnv, reasoning-gym, NeMo Gym resources servers | yes |
| `rollout` | the environment calls your model endpoint | Verifiers (legacy and v1) | no, `UnsupportedCapability` |
| `task` | an agent works in a sandbox; tests grade the end state | Harbor (Terminal-Bench, 80+ adapted benchmarks) | no, `UnsupportedCapability` |

`rollout()` is universal. `session()` exists only where it is true.

## Policies

There are exactly two kinds of policy, because environments consume exactly two kinds:

```python
lambda obs: obs.data["legal_actions"][0]            # ActionPolicy: Observation -> native action
nenyax.Endpoint(base_url, model)                    # ChatModel: an OpenAI-compatible server
nenyax.FunctionModel(lambda messages: "<answer>4</answer>")   # ChatModel: plain Python
nenyax.NativeAgent("oracle")                        # a format's own agent (e.g. Harbor's oracle/nop)
```

A `ChatModel` drives text step environments directly, and rollout/task environments through the
recording server. You never write a per-format adapter.

## Conformance: measured, not claimed

```
$ nenyax check reasoning_gym:basic_arithmetic
reasoning_gym:basic_arithmetic  [step]  →  tier: AUDITED
  ✔ connectable manifest
  ✔ connectable session_opens
  ✔ evaluable   rollout_completes
  ✔ trainable   reproducible
  ✔ trainable   learning_signal
  ✔ audited     verifier_discriminates
```

| Tier | Meaning |
|---|---|
| `connectable` | loads, the manifest is valid, an episode can start |
| `evaluable` | a full episode completes with a finite judgment |
| `trainable` | the same seed recreates the situation (needed for group-based RL such as GRPO), and the model path is recorded |
| `audited` | the verifier discriminates: a reference policy beats a null policy on every sampled task |

A check that cannot run is *skipped*, and a skip blocks its tier: an unverifiable verifier is not an audited one.

## Verified formats

Every row below was run end to end: by the integration suite (`tests/test_integration.py`), by the
real-model experiment (`examples/one_model_every_format.py`, Qwen2.5-1.5B served by mlx-lm), or both.

| Driver | Source | Mode | Highest tier reached | Notes |
|---|---|---|---|---|
| `reasoning_gym` | reasoning-gym 0.1.25 | step | audited | 106 procedural datasets, native verifiers |
| `gymnasium` | Gymnasium 1.3 | step | trainable | no oracle exists, so no audit |
| `openenv` | OpenEnv 0.6 | step | evaluable (public Spaces), trainable (seeded servers) | one generic client for every OpenEnv server |
| `nemo_gym` | NeMo Gym 0.6 | step | audited | `mcqa` and `reasoning_gym` resources servers, over HTTP |
| `verifiers` | Verifiers 0.3.1 | rollout | audited | legacy `SingleTurnEnv`/`ToolEnv` and v1 tasksets; Environments Hub `gsm8k` |
| `harbor` | Harbor 0.23 | task | audited | `terminus-2` wraps any ChatModel; `oracle`/`nop` audit the tests |

## Define, judge, improve

**Define** an environment in a few lines, from a dataset and a scorer, or from state functions:

```python
env = nenyax.define("capitals",
    tasks=[{"prompt": "Capital of Kenya? Use <answer></answer>.", "answer": "Nairobi"}, ...],
    score=lambda task, reply: float(task["answer"] in reply))
```

**Judge** any environment your own way, without touching it: exact match, regex or format rules,
Python functions, an LLM with a plain-language rubric, or weighted combinations:

```python
env = nenyax.with_judge(nenyax.load("verifiers:gsm8k"), nenyax.judges.weighted({
    nenyax.judges.EnvJudge(): 1.0, nenyax.judges.Regex(r"<answer>"): 0.1}))
```

**Improve** with any learner. A learner is anything with `policy()` and `update(groups)`:

| Learner | Algorithm | Works with |
|---|---|---|
| `incontext` | curates worked examples from the model's own best episodes; no weight changes | any chat model, including closed APIs |
| `grpo` | GRPO: group mean/std advantages, clipped ratio | local open-weight models (PyTorch: CUDA, MPS, CPU) |
| `dr_grpo` | Dr. GRPO: no std normalisation, token-level loss | ″ |
| `dapo` | DAPO: clip-higher (0.2, 0.28), token-level loss | ″ |
| `rloo` | RLOO: leave-one-out baseline | ″ |
| `reinforce` / `reinforce_pp` | REINFORCE with a batch baseline / REINFORCE++-style normalisation | ″ |
| `gspo` | GSPO: sequence-level importance ratio | ″ |
| `pg` | every knob: `advantage`, `loss` (reinforce, ppo, gspo), `aggregate`, `clip`, `epochs`, `kl` | ″ |
| `rft` | rejection-sampling fine-tuning (STaR / expert iteration) | ″ |
| `dpo` | DPO on best-vs-worst pairs from each group | ″ |
| `tinker` | hosted LoRA on Thinking Machines Tinker; the environment stays local | your `TINKER_API_KEY` |
| yours | anything with `policy()` and `update(groups)` | `nenyax.learners` entry point or `module:Class` |

The weight-updating learners are verified for correctness (exact advantage maths, real gradient
steps, conformance) but not yet benchmarked for learning at scale: treat them as experimental.

```python
result = nenyax.train(env, learner, rounds=10, group_size=8, batch=8)   # held-out eval, gated
print(result.render())
```

`train()` holds out tasks, evaluates after every update, and reverts updates that make held-out
performance worse. It also reports the fraction of groups that carried any learning signal.

## Configure everything

No user code needs to import a vendor class. Every piece is chosen by name (or `module:Class`), in
Python or in a config file:

```toml
[env]
uri = "reasoning_gym:basic_arithmetic?size=200"

[model]                     # any provider: vllm, ollama, openrouter, together, ...
provider = "vllm"
name = "Qwen/Qwen3-8B"

[learner]                   # incontext | grpo | tinker | my_pkg.learners:MyLearner
use = "incontext"

[train]
rounds = 10
group_size = 8
```

```bash
nenyax train --config run.toml
```

Example configs are in [examples/configs](examples/configs): local in-context, local GRPO, hosted
Tinker, and Terminal-Bench.

## Port out: serve anything as OpenEnv

```bash
nenyax serve "reasoning_gym:basic_arithmetic?size=50" --port 8000
```

Any step-mode environment becomes a standard OpenEnv server (`/ws`, `/reset`, `/step`, `/schema`),
so OpenEnv clients and the trainers built on them can use it unchanged. The episode's judgment rides
along in the final observation's metadata. The integration suite checks the full loop: reasoning-gym,
then OpenEnv server, then back into Nenyax.

## Sandboxes: isolation only where it is needed

The model, the agent loop and the environment are separate pieces, and only an environment whose
actions are dangerous needs a sandbox. Each environment declares the minimum `isolation` it needs
(`none` < `process` < `jail` < `container` < `microvm`), and Nenyax picks the lightest ready
backend that is safe enough:

```python
from nenyax import sandbox

backend = sandbox.select("container")            # None when isolation="none": no sandbox at all
with sandbox.WarmPool(backend, setup=["pip install numpy"], size=8) as pool:
    with pool.acquire() as sb:                   # fresh, identical state per episode (via fork)
        print(sb.exec("python3 -c 'print(6*7)'").stdout)
```

| Backend | Isolation | Enforces | Fork | Status |
|---|---|---|---|---|
| `process` | process | CPU time, timeout (and memory on Linux) | disk | built in, conformance-tested |
| `docker` (or Podman) | container | CPU, memory, network, filesystem, timeout | disk (`commit`) | built in, conformance-tested |

**Bring your own sandbox:** any provider (hosted or self-hosted) is one `SandboxBackend` class with
your key, published as a plugin (`nenyax new-plugin sandbox <name>`). See [SECURITY.md](SECURITY.md)
for what each isolation level does and does not protect against.

The contract is the one every provider converges on: `create · exec · files · destroy`, with
optional `fork` and `url(port)`. Hosted or proprietary providers plug in with your own key through
the `nenyax.sandboxes` entry point and must pass the same suite
(`nenyax sandbox-check <name>`). See
[docs/writing-a-sandbox-backend.md](docs/writing-a-sandbox-backend.md).

## Integrations

Every integration is one of a few kinds, with one small interface each. Hosted services use your
own key (Nenyax never stores it), and `nenyax integrations` shows what each one still needs.

| Kind | Built in | How it plugs in |
|---|---|---|
| Model providers | OpenAI, Anthropic, Gemini, Mistral, OpenRouter, Together, Fireworks, Groq, Cerebras, DeepInfra, HF Inference Providers, Prime Inference; local vLLM, SGLang, Ollama, LM Studio, mlx-lm, llama.cpp | `integrations.endpoint("together", "Qwen/Qwen3-8B")` |
| Environment hubs | Hugging Face (OpenEnv Spaces), Prime Environments Hub, Harbor registry, all searchable without a key | `nenyax search prime math`, then `integrations.fetch(hit)` |
| Telemetry | OpenTelemetry (GenAI conventions, so Langfuse, Phoenix and Jaeger can ingest it), MLflow, Weights & Biases | `nenyax.run(..., callbacks=[...])` |
| Exports | JSONL, Parquet, Hugging Face datasets and Hub, as raw, SFT or DPO rows | `integrations.export.to_rows(trajs, "dpo")` |
| Sandboxes | process, docker | see above |

## Extend anything: plugins

Every piece is a plugin kind with one small interface, an entry-point group and a conformance test:
**driver, sandbox, provider, hub, learner, judge and telemetry** (40 built in). Anyone,
companies included, can add one without touching Nenyax:

```bash
nenyax new-plugin sandbox acme      # a package that passes its conformance test on day one
nenyax plugins                      # everything installed, built-in and third-party
nenyax plugins --kinds              # the extension points
```

Publish the package, and users pick it by name (`use = "acme"`). To be listed publicly, open a
PR adding it to [`registry/plugins.toml`](registry/plugins.toml); CI installs it and checks the
entry point. See [CONTRIBUTING.md](CONTRIBUTING.md) for both routes: your own package, or a PR
into core.

## CLI

```bash
nenyax drivers                      # what is installed
nenyax info  <uri>                  # manifest
nenyax check <uri>                  # conformance tier
nenyax run   <uri> --policy endpoint --base-url http://localhost:8000/v1 --model M -n 20 --out t.jsonl
nenyax serve <uri> --port 8000      # expose as OpenEnv
nenyax try   <uri> --provider ollama --model qwen3   # one episode, readable replay
nenyax train <uri> --learner incontext --provider vllm --model M   # or: --config run.toml
nenyax integrations                 # every integration and what it still needs
nenyax plugins [--kinds|--registry] # installed plugins, extension points, public registry
nenyax new-plugin <kind> <name>     # scaffold a publishable plugin
nenyax search harbor terminal-bench # find environments on huggingface / prime / harbor
nenyax sandboxes                    # sandbox backends and what each needs (software, keys)
nenyax sandbox-check docker         # verify a backend delivers what it claims
```

## Writing a driver

A driver is one class. Third-party packages register it through an entry point, with no change to Nenyax:

```toml
[project.entry-points."nenyax.drivers"]
mydriver = "my_pkg.nenyax_driver:MyDriver"
```

See [docs/writing-a-driver.md](docs/writing-a-driver.md). Prove it works with
`nenyax.testing.assert_conforms(env, "trainable")`.

## Status

Alpha (`0.1.0a1`, contract `0.1`), tested on macOS and Linux arm64 with Python 3.12. The contract
is described in [docs/protocol.md](docs/protocol.md), what changed in [CHANGELOG.md](CHANGELOG.md),
and what we learned building it in [docs/learnings.md](docs/learnings.md). Known gaps:

- NeMo Gym *agent* servers (multi-turn, tool-calling rows) are not driven yet; those rows are rejected, not mis-scored.
- Harbor wraps chat models with `terminus-2`; other Harbor agents run as `NativeAgent`.
- Verifiers v1 starts its interception proxy once per rollout. It is correct but slow; pooling is planned.
- Token ids are captured when the upstream returns them (vLLM, SGLang). Nenyax never re-tokenizes.

## License

Apache-2.0
