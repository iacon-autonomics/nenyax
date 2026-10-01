# Trainers

Nenyax supplies the environment, the rewards and the evaluation; you pick the trainer. Every
trainer below turns a Nenyax environment into that trainer's native job, so you keep the
framework you already use and swap it without touching the environment.

| Trainer | What it runs | Rewards come from | Status |
|---|---|---|---|
| `nenyax` | the built-in learners: in-context, GRPO, DAPO, RLOO, RFT, DPO, ... | the environment | ready |
| `trl` | TRL `GRPOTrainer` (or `DPOTrainer` on preference pairs) | a reward function calling the environment's verifier | beta |
| `unsloth` | GRPO through TRL with Unsloth 4-bit loading and LoRA | same as `trl` | beta |
| `verl` | `verl.trainer.main_ppo` (GRPO, PPO, RLOO, REINFORCE++) | verl `custom_reward_function` backed by the verifier | beta |
| `prime-rl` | prime-rl on a verifiers environment that wraps the Nenyax environment | the verifiers rubric = the Nenyax verifier | beta |
| `openrlhf` | `openrlhf.cli.train_ppo_ray` | a remote reward server Nenyax runs | beta |
| `tinker` | Thinking Machines Tinker, hosted LoRA (`TINKER_API_KEY`) | the environment | beta |

**ready** means `run()` has been exercised for real. **beta** means `prepare()` generates the
job and is tested, and the reward path is tested against real environments, but the job has not
yet been run end to end on a GPU against the real trainer.

## Use one

```python
import nenyax
from nenyax import trainers

env = nenyax.load("reasoning_gym:basic_arithmetic?size=500")
trl = trainers.get("trl")
print(trl.available())                     # (False, "needs trl ...; needs 1 CUDA GPU") on a laptop
job = trl.prepare(env, "Qwen/Qwen2.5-0.5B-Instruct", {"max_steps": 200}, "./job")
print(job.shell)                           # the exact command; run it on any GPU box
trl.run(job, events=print)                 # or here, streaming NENYAX_METRIC lines as events
```

`prepare()` always works, even without the trainer installed, so a job can be prepared on your
laptop and launched on compute elsewhere (the platform's compute plugs do exactly this).

## What a job contains

* `train.jsonl` / `eval.jsonl` (or verl's parquet): the environment's tasks, split
  deterministically so trainers never see the held-out tasks.
* `nenyax_bridge.py`: rebuilds the environment from its URI and scores completions with its own
  verifier. The trainer process needs `pip install nenyax` and the environment's driver.
* The trainer's own entry point: `train_trl.py`, verl Hydra overrides, prime-rl `rl.toml` plus
  `nenyax_verifiers_env.py`, or OpenRLHF flags plus a `reward_server.py` sidecar.

The environment must be rebuildable in another process: load it with `nenyax.load(uri)`, or pass
`config={"env": "<uri>"}`. Environments defined inline in Python need `{"env": {"use":
"module:factory"}}`.

## In a config or a pipeline

Every trainer is also a learner, so the gate and held-out evaluation are the usual ones:

```toml
[learner]
use = "trl"
model = "Qwen/Qwen2.5-0.5B-Instruct"
max_steps = 200
eval_model = { provider = "vllm", name = "Qwen/Qwen2.5-0.5B-Instruct", base_url = "http://localhost:8000/v1" }
```

`eval_model` is the endpoint that serves the model for the before/after evaluation; after training
Nenyax evaluates the checkpoint through the same server (or `eval_after`, if given).

## Any environment as a verifiers environment

`nenyax.trainers.prime_rl.to_verifiers(env)` turns any Nenyax step environment into a legacy
`verifiers` Environment: single-turn ones become a `SingleTurnEnv`, multi-turn ones a
`MultiTurnEnv` that steps a live Nenyax session per assistant turn. That makes every Nenyax
environment usable by prime-rl, `vf-eval`, and anything else built on verifiers.
