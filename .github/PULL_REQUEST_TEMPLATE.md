## What this adds or fixes

<!-- One or two sentences. Link an issue if there is one. -->

## Kind

- [ ] Core (contract, CLI, conformance)
- [ ] Built-in plugin: driver / sandbox / provider / hub / learner / judge / telemetry
- [ ] Registry listing (`registry/plugins.toml`), third-party plugin in its own package
- [ ] Docs / examples

## Checklist

- [ ] It passes its conformance test (`nenyax.testing.assert_*_conforms`), run against the **real** package or service where possible
- [ ] Capabilities and claims are honest: nothing is declared that the test doesn't check
- [ ] Vendor SDKs are imported lazily, and credentials only come from env vars or explicit options, never stored
- [ ] `ruff check`, `ruff format --check` and `pytest` pass
- [ ] If this touches the contract (`nenyax.types`), the change is backward compatible or called out here
- [ ] README integrations table / docs updated with the status actually measured

## How I tested it

<!-- Commands and output. For keyed services, say whether it ran live. -->
