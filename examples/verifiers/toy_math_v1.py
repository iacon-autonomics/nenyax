import re

import verifiers.v1 as vf


class ToyData(vf.TaskData):
    answer: str


class ToyTask(vf.Task[ToyData]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        m = re.findall(r"<answer>(.*?)</answer>", trace.last_reply or "")
        return 1.0 if m and m[-1].strip() == self.data.answer else 0.0

    @vf.metric
    async def reply_len(self, trace: vf.Trace) -> float:
        return float(len(trace.last_reply or ""))


class ToyTaskset(vf.Taskset[ToyTask, vf.TasksetConfig]):
    def load(self):
        rows = [("What is 2+2?", "4"), ("What is 3+5?", "8"), ("What is 10-7?", "3")]
        for i, (q, a) in enumerate(rows):
            yield ToyTask(
                ToyData(
                    idx=i, prompt=q, system_prompt="Answer in <answer></answer> tags.", answer=a
                ),
                self.config.task,
            )


__all__ = ["ToyTaskset"]
