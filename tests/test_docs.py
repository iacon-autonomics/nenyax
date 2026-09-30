"""Keep docs/writing-a-driver.md honest: its example must actually conform."""

from pathlib import Path

DOC = Path(__file__).resolve().parent.parent / "docs" / "writing-a-driver.md"


def test_writing_a_driver_example_reaches_trainable():
    import re

    blocks = re.findall(r"```python\n(.*?)```", DOC.read_text(), re.S)
    namespace: dict = {}
    exec(blocks[0], namespace)  # the driver
    import nenyax
    from nenyax.testing import assert_conforms

    nenyax.register(namespace["GuessDriver"]())
    env = nenyax.load("guess:1-10")
    assert_conforms(env, "trainable")
    guesses = iter(range(1, 11))
    traj = env.rollout(lambda obs: str(next(guesses)), seed=4)  # secret = 5
    assert traj.score == 1.0 and len(traj.steps) == 5
