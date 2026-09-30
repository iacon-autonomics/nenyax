# Writing a sandbox backend

A sandbox backend gives Nenyax somewhere to run untrusted actions. The contract is the same set of
operations every provider already offers:

| Operation | Required | Notes |
|---|---|---|
| `create(spec) -> Sandbox` | yes | honour `SandboxSpec` (image, cpus, memory, network, env, ports) |
| `exec(cmd, timeout_s, cwd, env) -> ExecResult` | yes | shell command; report `timed_out` honestly |
| `write_file` / `read_file` | yes | bytes in, bytes out |
| `destroy()` | yes | idempotent |
| `fork() -> Sandbox` | optional | declare `fork="disk"` or `"memory"` |
| `url(port) -> str` | optional | declare `ports=True` |

Declare what you really do in `BackendInfo`. The conformance suite checks every claim.

## A hosted provider with your own key

Keys are never stored by Nenyax. They come from an environment variable, or from an option passed
at construction time.

```python
from nenyax.sandbox import (BackendInfo, ExecResult, Isolation, Sandbox, SandboxBackend,
                            SandboxSpec)


class AcmeSandbox(Sandbox):
    def __init__(self, client, spec: SandboxSpec):
        self.client, self.spec = client, spec
        self.id = client.create(image=spec.image, cpu=spec.cpus, memory_mb=spec.memory_mb,
                                internet=spec.network, env=spec.env)

    def exec(self, command, *, timeout_s=None, cwd=None, env=None) -> ExecResult:
        r = self.client.run(self.id, command, timeout=timeout_s, cwd=cwd, env=env)
        return ExecResult(r.exit_code, r.stdout, r.stderr, r.timed_out, r.seconds)

    def write_file(self, path, data):
        self.client.upload(self.id, path, data if isinstance(data, bytes) else data.encode())

    def read_file(self, path):
        return self.client.download(self.id, path)

    def fork(self):                      # only if the provider can snapshot/branch
        child = AcmeSandbox.__new__(AcmeSandbox)
        child.client, child.spec, child.id = self.client, self.spec, self.client.branch(self.id)
        return child

    def destroy(self):
        self.client.delete(self.id)


class AcmeBackend(SandboxBackend):
    name = "acme"
    credentials = ("ACME_API_KEY",)      # bring your own key
    info = BackendInfo(
        isolation=Isolation.MICROVM,
        enforces=frozenset({"cpu", "memory", "network", "filesystem", "timeout"}),
        fork="memory",
    )

    def create(self, spec=None):
        import acme_sdk                  # import the vendor SDK lazily

        client = acme_sdk.Client(api_key=self.credential("ACME_API_KEY"))
        return AcmeSandbox(client, spec or SandboxSpec())
```

Register it from your own package, with no change to Nenyax:

```toml
[project.entry-points."nenyax.sandboxes"]
acme = "acme_nenyax:AcmeBackend"
```

Use it:

```python
from nenyax import sandbox

backend = sandbox.get("acme")                           # reads ACME_API_KEY
backend = sandbox.get("acme", acme_api_key="sk-...")    # or pass the key explicitly
backend = sandbox.select("microvm", fork="memory", prefer=["acme"])
```

`sandbox.get` and `sandbox.select` report exactly what is missing, e.g. `set ACME_API_KEY`.

## Prove it

```python
from nenyax.testing import assert_sandbox_conforms

def test_acme():
    assert_sandbox_conforms(AcmeBackend())
```

Or run `nenyax sandbox-check acme`. The suite creates a sandbox and checks `exec`, exit codes,
file round trips, timeouts, network blocking, filesystem containment and fork isolation, then
destroys it. A claimed guarantee that does not hold is a failure.

## Building on top

Backends are plain objects, so anything can be composed around them without subclassing Nenyax:

- `WarmPool(backend, setup=[...], size=N)`: set up once, then hand out forks.
- Wrap a backend to add logging, metering, retries or quotas. It is still a `SandboxBackend`.
- `select()` accepts `prefer=[...]`, so routing policy (cost, region, provider) stays in your code.
