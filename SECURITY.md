# Security

Nenyax runs actions chosen by models: shell commands, code, tool calls. How dangerous that is
depends on the environment, so Nenyax makes isolation an explicit, declared, tested property
rather than an assumption.

## Threat model in one table

| Isolation | Backend | Protects the host from | Does **not** protect against |
|---|---|---|---|
| `none` | no sandbox | nothing (fine for pure functions: math, `/verify`-style scoring) | anything the action does |
| `process` | `process` | runaway CPU time, fork-heavy loops (fd limits), memory on Linux | reading or writing any file your user can, network access, memory on macOS |
| `container` | `docker` / `podman` | filesystem access, network (off by default), CPU, memory, process count | kernel exploits (the host kernel is shared), disk filling (no quota by default) |
| `microvm` | bring-your-own-key backends, or future Firecracker/Kata | kernel-level isolation | provider-specific limits; read their docs |

Rules we follow and test (`nenyax sandbox-check <backend>`):

- A backend declares what it **enforces**. Every declared guarantee is attacked by the
  conformance suite (network egress, host-file reads, timeouts, fork isolation). A guarantee that
  does not hold is a failure.
- Output is capped per stream (1 MiB by default), so a flood of output cannot exhaust the
  controller's memory.
- Host environment variables are not passed into sandboxes. Only `SandboxSpec.env` is.
- Docker sandboxes run with `--network none` unless networking is requested, plus CPU, memory
  and process-count limits.

**Use the right level:** untrusted model-written code belongs in `container` at minimum, and in
`microvm` when the model or the task is adversarial. `process` is for code you would run on your
own machine anyway.

## Credentials

- Nenyax never stores credentials. Keys come from environment variables or explicit arguments,
  and integrations only report *which* variable is missing.
- Nenyax never sends telemetry anywhere unless you add a telemetry callback yourself.
- Environment servers (`nenyax serve`) bind to `127.0.0.1` by default.

## Known limits (0.1.0a1)

- The `process` backend does not limit process count, disk, or (on macOS) memory. Never use it
  for untrusted code.
- Docker enforces no disk quota in the default configuration.
- A fork bomb inside a Docker sandbox is contained (`--pids-limit`) but leaves that sandbox
  unusable. Discard it; warm pools already hand out fresh ones.
- Remote environments (OpenEnv servers, Hugging Face Spaces, NeMo Gym servers) run code you do
  not control. Treat their observations as untrusted input.

## Reporting a vulnerability

Please do not open a public issue. Email **dev@iaconautonomics.com** with a description and a
reproduction. We aim to acknowledge within three working days.
