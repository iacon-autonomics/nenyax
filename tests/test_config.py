import json

import pytest

import nenyax
from nenyax import FunctionModel, config
from nenyax.config import ConfigError
from nenyax.learners import InContextLearner


def make_env(size: int = 6):
    """An env factory referenced from configs as 'test_config:make_env'."""
    tasks = [{"prompt": f"Say {i} in answer tags.", "answer": str(i)} for i in range(size)]
    return nenyax.define(
        "say", tasks=tasks, score=lambda t, r: float(r.strip() == f"<answer>{t['answer']}</answer>")
    )


def test_every_kind_is_built_by_name():
    judge = config.build(
        "judge",
        {
            "use": "weighted",
            "parts": [
                {"use": "env", "weight": 1.0},
                {"use": "regex", "pattern": "<answer>", "weight": 0.5},
            ],
        },
    )
    assert type(judge).__name__ == "Weighted" and len(judge.parts) == 2
    assert config.build("model", {"provider": "vllm", "name": "m"}).base_url.endswith(":8000/v1")
    assert config.build("sandbox", "process").name == "process"
    assert type(config.build("judge", "nenyax.judges:ExactMatch")).__name__ == "ExactMatch"


def test_unknown_names_list_the_alternatives():
    with pytest.raises(ConfigError, match="registered"):
        config.build("learner", "definitely-not-a-learner")


def test_env_spec_by_uri_factory_and_with_judge():
    assert config.build_env("reasoning_gym:leg_counting?size=2").manifest.num_tasks == 2
    env = config.build_env({"use": "test_config:make_env", "size": 4, "judge": "env"})
    assert env.manifest.num_tasks == 4 and env.id.endswith("+judge")


def test_run_from_a_json_file(tmp_path):
    cfg = {
        "env": {"use": "test_config:make_env", "size": 8},
        "judge": {"use": "regex", "pattern": "^<answer>\\d+</answer>$"},
        "train": {"rounds": 1, "group_size": 1, "batch": 2},
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(cfg))
    learner = InContextLearner(
        FunctionModel(lambda m: f"<answer>{m[-1].content.split()[1]}</answer>")
    )
    result = config.run({**config.read(path), "learner": learner})
    assert result.baseline == 1.0 and len(result.history) == 1


def test_example_configs_parse():
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "examples" / "configs"
    for path in root.glob("*.toml"):
        cfg = config.read(path)
        assert "env" in cfg and "learner" in cfg, path.name
