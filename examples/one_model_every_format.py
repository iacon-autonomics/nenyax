"""One model, one Endpoint, every environment format.

Point this at any OpenAI-compatible server (vLLM, SGLang, mlx-lm, Ollama, a hosted API)::

    NENYAX_BASE_URL=http://127.0.0.1:8000/v1 NENYAX_MODEL=Qwen/Qwen3-8B \\
        python examples/one_model_every_format.py

Optional services add more rows:

    NENYAX_NEMO_GYM_URL / NENYAX_NEMO_GYM_DATA   a NeMo Gym reasoning_gym resources server
    NENYAX_OPENENV_URL                           a Nenyax-exported OpenEnv server (text actions)
    Docker running                               the Harbor task
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import nenyax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "verifiers"))  # the toy Verifiers environments live here

endpoint = nenyax.Endpoint(
    base_url=os.environ.get("NENYAX_BASE_URL", "http://127.0.0.1:8000/v1"),
    model=os.environ.get("NENYAX_MODEL", "default"),
    api_key=os.environ.get("NENYAX_API_KEY", "EMPTY"),
    default_params={"temperature": 0.0, "max_tokens": 512},
)
EPISODES = int(os.environ.get("NENYAX_EPISODES", "5"))

targets: list[tuple[str, dict]] = [
    ("reasoning_gym:basic_arithmetic?size=20&seed=7", {}),
    ("reasoning_gym:leg_counting?size=20&seed=7", {}),
    ("verifiers:toy_math_legacy", {}),
    ("verifiers:toy_math_v1", {}),
]
try:
    import gsm8k  # noqa: F401  (vf-install primeintellect/gsm8k)

    targets.append(("verifiers:gsm8k", {}))
except ImportError:
    pass
if os.environ.get("NENYAX_NEMO_GYM_URL"):
    targets.append(
        (
            f"nemo_gym:{os.environ['NENYAX_NEMO_GYM_URL']}",
            {"data": os.environ["NENYAX_NEMO_GYM_DATA"]},
        )
    )
if os.environ.get("NENYAX_OPENENV_URL"):
    targets.append(
        (f"openenv:{os.environ['NENYAX_OPENENV_URL']}", {"seeded": True, "text_action": "value"})
    )
if os.environ.get("NENYAX_HARBOR", "1") == "1":
    targets.append((f"harbor:{HERE / 'harbor' / 'hello'}", {}))

print(f"model: {endpoint.model} @ {endpoint.base_url}\n")
header = ("format", "mode", "environment", "score", "ok", "calls", "time")
print("{:<18} {:<8} {:<34} {:>6} {:>5} {:>6} {:>7}".format(*header))
for uri, options in targets:
    t0 = time.perf_counter()
    try:
        env = nenyax.load(uri, **options)
    except Exception as e:
        print(f"{uri:<60} load failed: {type(e).__name__}: {e}")
        continue
    n = 1 if env.mode is nenyax.Mode.TASK else EPISODES
    trajs = list(nenyax.run(env, endpoint, n=n))
    ok = [t for t in trajs if t.ok]
    mean = sum(t.score for t in ok) / len(ok) if ok else float("nan")
    calls = sum(len(t.model_calls) for t in trajs)
    name = env.manifest.name[:34]
    print(
        f"{env.manifest.source_format:<18} {env.mode.value:<8} {name:<34} {mean:6.2f} "
        f"{len(ok):>2}/{len(trajs):<2} {calls:>6} {time.perf_counter() - t0:6.1f}s"
    )
    for t in trajs:
        if t.error:
            print(f"    error: {t.error[:160]}")
