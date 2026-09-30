# Writing a driver

A driver translates one native environment format into the Nenyax contract. You write one class
for the environment and one for the driver.

## 1. Pick the mode

- The caller can step the environment? Subclass `StepEnvironment` and implement `session()`.
- The environment calls a model itself? Subclass `ModelDrivenEnvironment` and implement `_run()`.

## 2. A step-mode driver

```python
import nenyax
from nenyax import Capabilities, Judgment, Manifest, Mode, Observation, Session, Step, StepEnvironment


class GuessSession(Session):
    def __init__(self, secret: int):
        self.secret, self.history, self.done = secret, [], False
        self.observation = Observation(text="Guess a number from 1 to 10.")

    def act(self, action):
        try:
            guess = int(str(action).strip())
        except ValueError:  # a policy may say anything; never crash on bad input
            guess = None
        hint = "not a number" if guess is None else ("higher" if guess < self.secret else "lower")
        won = guess == self.secret
        step = Step(action=action, observation=Observation(text=hint),
                    reward=float(won), terminated=won)
        self.history.append(step)
        self.done = won
        return step

    def judge(self):
        return Judgment(score=float(self.done), source="verifier", judge="guess@1")


class GuessEnv(StepEnvironment):
    default_max_steps = 10

    def __init__(self):
        self.manifest = Manifest(id="guess:1-10", name="guess", source_format="guess",
                                 mode=Mode.STEP, capabilities=Capabilities(text=True))

    def session(self, *, task=None, seed=None):
        return GuessSession(secret=(seed or 0) % 10 + 1)


class GuessDriver(nenyax.Driver):
    name, format = "guess", "Guess"

    def load(self, target, **options):
        return GuessEnv()
```

## 3. A model-driven driver

Implement `_run(endpoint, *, task, seed, max_steps)`. Hand `endpoint.base_url` and
`endpoint.model` to your framework as its OpenAI-compatible model, run one native episode, and
return `(Judgment, messages, metadata)`. Nenyax starts and stops the recording server around it,
so `trajectory.model_calls` is filled in for you. If your framework ships its own agents, override
`_run_native(agent, ...)` too.

## 4. Enable the audit tier

Implement `reference_policy(task)` (something expected to succeed, such as an oracle or a gold
answer) and `null_policy(task)` (something expected to fail). Without them your environment tops
out at `trainable`, which is the honest answer.

For model-driven environments whose agent expects a specific reply format, implement
`probe_policy()`: a cheap `ChatModel` that speaks that format, so conformance can prove the model
path works.

## 5. Register it

```toml
[project.entry-points."nenyax.drivers"]
guess = "my_pkg.guess:GuessDriver"
```

or, at runtime, `nenyax.register(GuessDriver())`.

## 6. Test it

```python
from nenyax.testing import assert_conforms

def test_guess_driver():
    assert_conforms(nenyax.load("guess:1-10"), "trainable")
```

## Rules

- Never over-claim a capability. If the format can't reset to a seed, set `resettable=False`.
- Never put privileged state (answers, hidden server state) in an `Observation`.
- Rollout failures belong in `Trajectory.error`, not in exceptions. Base classes already do this.
- Import the native package lazily, and use `Driver.require()` for a helpful install hint.
