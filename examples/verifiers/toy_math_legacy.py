"""A tiny legacy Verifiers environment used by Nenyax's integration tests."""

import verifiers as vf
from datasets import Dataset


def load_environment(**kwargs):
    ds = Dataset.from_list(
        [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is 3+5?", "answer": "8"},
            {"question": "What is 10-7?", "answer": "3"},
        ]
    )
    parser = vf.XMLParser(["answer"], answer_field="answer")

    def correct(completion, answer, **kw) -> float:
        return 1.0 if parser.parse_answer(completion) == answer else 0.0

    rubric = vf.Rubric(funcs=[correct], parser=parser)
    return vf.SingleTurnEnv(
        eval_dataset=ds,
        system_prompt="Answer in <answer></answer> tags.",
        parser=parser,
        rubric=rubric,
        env_id="toy_math_legacy",
    )


def load_tool_environment():
    ds = Dataset.from_list([{"question": "Use the add tool to compute 17+25.", "answer": "42"}])

    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    parser = vf.XMLParser(["answer"], answer_field="answer")

    def correct(completion, answer, **kw) -> float:
        return 1.0 if parser.parse_answer(completion) == answer else 0.0

    return vf.ToolEnv(
        eval_dataset=ds,
        tools=[add],
        max_turns=4,
        parser=parser,
        rubric=vf.Rubric(funcs=[correct], parser=parser),
        env_id="toy_tools",
    )
