# The Nenyax contract, v0.1

Status: draft. This describes the data every environment exchanges regardless of native format,
and the obligations drivers have. The normative definitions are the Pydantic models in
`src/nenyax/types.py`; this document explains them.

## 1. Design rules

1. **Translate, don't replace.** A driver maps a native format onto the contract. It never asks
   environment authors to rewrite anything.
2. **Never over-claim.** `Capabilities` states what an environment can *actually* do. If it
   cannot be stepped, recreated or forked, the driver says so, and callers get
   `UnsupportedCapability` instead of a fake.
3. **One universal operation.** `rollout(policy) -> Trajectory` works for every mode. Finer
   control (`session()`) exists only where it is true.
4. **Everything is recorded.** Every model call an environment makes goes through a recording
   endpoint, so any environment can feed a trainer.
5. **Tiers are measured.** Conformance is a test suite, not a badge you declare yourself.

## 2. Identity: `Manifest`

| Field | Meaning |
|---|---|
| `id` | canonical URI, `<driver>:<target>` |
| `name`, `version`, `license`, `description` | descriptive |
| `source_format` | native format translated from (`verifiers/v1`, `harbor`, ...) |
| `mode` | `step`, `rollout` or `task` (section 3) |
| `capabilities` | section 4 |
| `action_schema`, `observation_schema` | JSON Schema, when the format provides one |
| `num_tasks` | size of the finite task set, if any |
| `extra` | format-specific metadata; never required by consumers |

## 3. Modes

Every native format maps to exactly one mode.

- **step**: the caller drives. `session(task, seed)` returns a `Session` with `observation`,
  `act(action) -> Step`, `done`, and `judge() -> Judgment`.
- **rollout**: the environment owns the episode loop and calls a model endpoint (an
  OpenAI-compatible chat-completions URL) that Nenyax provides.
- **task**: a sandboxed task. An agent works inside it, and tests grade the final state after the
  agent stops. Nenyax plugs chat models in behind a format agent, or runs the format's own agents
  via `NativeAgent`.

## 4. Capabilities

| Field | Type | Meaning |
|---|---|---|
| `resettable` | bool | the same situation can be recreated from a seed or task id |
| `branchable` | bool | a live episode can be forked |
| `max_parallel` | int or null | concurrency bound, if known |
| `participants` | int | agents per episode |
| `text` | bool | observations render as text and actions accept text (LLM-drivable) |
| `reward_timing` | `per_step` or `terminal` | when reward arrives |
| `real_world` | bool | actions have real-world consequences |
| `risk` | `none`, `reversible` or `irreversible` | worst-case effect of an action |
| `requires` | list | `docker`, `network`, `gpu`, `model_endpoint`, `api_key` |
| `token_capture` | `none`, `proxy` or `native` | how model tokens are captured |

## 5. Policies

- `ActionPolicy`: `Observation -> native action`.
- `ChatModel`: `messages -> assistant Message` (`Endpoint`, `FunctionModel`, or your own class).
- `NativeAgent(name)`: an agent that ships with the format.

Routing rules:

| | ActionPolicy | ChatModel | NativeAgent |
|---|---|---|---|
| step, text | direct | observations rendered as chat turns | n/a |
| step, non-text | direct | `PolicyMismatch` | n/a |
| rollout / task | wrapped as `FunctionModel` over messages | served via the recording server | the format's own agent |

## 6. Records

- `Observation{data, text?, messages?, info}`: only what the acting policy may see. Privileged
  state (answers, server state) never appears here.
- `Step{action, observation, reward, terminated, truncated, info}`.
- `ModelCall{request, response, error, latency_s, prompt_token_ids?, completion_token_ids?, completion_logprobs?}`.
- `Judgment{score (finite), passed?, components, source, judge, details}`, where `source` is one
  of `env`, `verifier`, `rubric`, `tests`, `human` or `external`, and `judge` names and versions
  the scorer.
- `Trajectory{env_id, mode, task, seed, initial_observation, steps, messages, model_calls, judgment, error, duration_s, metadata}`:
  the unit trainers consume. A rollout that fails is returned with `error` set; it is never raised.

## 7. The recording model server

For rollout and task modes, Nenyax starts an OpenAI-compatible server for each episode, addressed
as `/s/<session>/v1`. It either *is* the model (backed by a `ChatModel`) or transparently proxies
to one (backed by an `Endpoint`). It supports `GET /v1/models`, `POST /v1/chat/completions`
(streaming and non-streaming) and tool calls in OpenAI's wire shape. Token ids and logprobs are
copied from upstream responses when present; Nenyax never re-tokenizes.

## 8. URIs

`<driver>:<target>[?key=value&...]`. Query values are parsed as JSON when possible. A bare local
path is routed to the first driver whose `detect(path)` returns true. Drivers are discovered from
built-ins plus the `nenyax.drivers` entry-point group.

## 9. Conformance tiers

| Tier | Checks |
|---|---|
| connectable | `manifest` validates; `session_opens` (step) or `tasks_listed` (rollout/task) |
| evaluable | `rollout_completes`: sampled episodes finish with a finite judgment |
| trainable | `reproducible`: same seed gives the same situation (skipped, and so blocking, if not `resettable`); `learning_signal`: step episodes record steps, and model-driven episodes record model calls with responses |
| audited | `verifier_discriminates`: a reference policy scores strictly above a null policy on every sampled task |

Tiers are cumulative. A skipped check blocks its tier.

## 10. OpenEnv interop

Nenyax is designed to sit next to OpenEnv, not compete with it:

- **in**: the `openenv` driver drives any OpenEnv server with one generic client.
- **out**: `nenyax serve` exposes any step-mode environment as an OpenEnv server. Actions are
  `{"value": <native>}`; observations carry `data`, `text` and `info`; the final observation's
  metadata carries `nenyax_judgment`.

Candidate contributions back to OpenEnv: the `Capabilities` block, the conformance tiers, and the
judgment provenance fields.
