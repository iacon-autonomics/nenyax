"""Thinking Machines Tinker: hosted LoRA training driven by Nenyax rollouts (bring your own key).

Nenyax samples through Tinker, keeps the exact tokens and logprobs it sampled, and sends
group-relative advantages back with Tinker's ``importance_sampling`` (or ``ppo``/``cispo``) loss.
The environment never leaves your machine; only tokens go to Tinker::

    learner = TinkerLearner("Qwen/Qwen3-8B")          # reads TINKER_API_KEY
    nenyax.train(env, learner, rounds=20, group_size=8)

Written against ``tinker`` 0.31's public SDK. Not yet exercised against the live service
(needs a key), so it is marked ``verified_live=False`` in ``nenyax integrations``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..types import Message, Trajectory
from .base import group_advantages, has_signal

_IDS = "nenyax_ids"


class _TinkerChat:
    def __init__(self, learner: TinkerLearner) -> None:
        self.learner, self.model = learner, learner.base_model

    def complete(self, messages: Sequence[Message], **params: Any) -> Message:
        import tinker

        lr = self.learner
        chat = [
            {"role": m.role, "content": m.content or ""}
            for m in messages
            if isinstance(m.content, str) or m.content is None
        ]
        prompt = lr.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True)
        if hasattr(prompt, "keys"):  # BatchEncoding (transformers >= 5) or dict
            prompt = prompt["input_ids"]
        prompt = [int(t) for t in prompt]
        result = lr.sampler.sample(
            prompt=tinker.ModelInput.from_ints(prompt),
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=int(params.get("max_tokens") or lr.max_tokens),
                temperature=lr.temperature,
                stop=[lr.tokenizer.eos_token_id] if lr.tokenizer.eos_token_id is not None else None,
            ),
        ).result()
        seq = result.sequences[0]
        text = lr.tokenizer.decode(seq.tokens, skip_special_tokens=True)
        return Message.model_validate(
            {
                "role": "assistant",
                "content": text,
                _IDS: {
                    "prompt": prompt,
                    "completion": list(seq.tokens),
                    "logprobs": list(seq.logprobs or []),
                },
            }
        )


class TinkerLearner:
    name = "tinker"
    credentials = ("TINKER_API_KEY",)
    verified_live = False

    def __init__(
        self,
        base_model: str,
        *,
        rank: int = 32,
        lr: float = 4e-5,
        loss_fn: str = "importance_sampling",
        max_tokens: int = 512,
        temperature: float = 1.0,
        service_client: Any = None,
    ) -> None:
        import tinker

        self.base_model, self.lr, self.loss_fn = base_model, lr, loss_fn
        self.max_tokens, self.temperature = max_tokens, temperature
        service = service_client or tinker.ServiceClient()
        self.trainer = service.create_lora_training_client(base_model=base_model, rank=rank)
        self.tokenizer = self.trainer.get_tokenizer()
        self.sampler = self.trainer.save_weights_and_get_sampling_client(name="nenyax-init")
        self.step = 0

    def policy(self) -> _TinkerChat:
        return _TinkerChat(self)

    @staticmethod
    def datum(prompt: list[int], completion: list[int], logprobs: list[float], adv: float) -> Any:
        """One training example in Tinker's shifted next-token layout; prompt positions masked."""
        import numpy as np
        import tinker

        tokens = prompt + completion
        pad = len(prompt) - 1
        lp = (list(logprobs) + [0.0] * len(completion))[: len(completion)]
        return tinker.Datum(
            model_input=tinker.ModelInput.from_ints(tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_numpy(np.array(tokens[1:], dtype=np.int64)),
                "logprobs": tinker.TensorData.from_numpy(
                    np.array([0.0] * pad + lp, dtype=np.float32)
                ),
                "advantages": tinker.TensorData.from_numpy(
                    np.array([0.0] * pad + [adv] * len(completion), dtype=np.float32)
                ),
            },
        )

    def update(self, groups: list[list[Trajectory]]) -> dict[str, float]:
        import tinker

        data = []
        for group, advs in zip(groups, group_advantages(groups), strict=True):
            for traj, adv in zip(group, advs, strict=True):
                if adv == 0.0:
                    continue
                for call in traj.model_calls:
                    msg = ((call.response or {}).get("choices") or [{}])[0].get("message") or {}
                    ids = msg.get(_IDS)
                    if ids and ids["completion"]:
                        data.append(
                            self.datum(ids["prompt"], ids["completion"], ids["logprobs"], adv)
                        )
        metrics = {"signal": has_signal(groups), "samples": float(len(data))}
        if not data:
            return metrics
        self.trainer.forward_backward(data, loss_fn=self.loss_fn).result()
        self.trainer.optim_step(adam_params=tinker.AdamParams(learning_rate=self.lr)).result()
        self.step += 1
        self.sampler = self.trainer.save_weights_and_get_sampling_client(name=f"nenyax-{self.step}")
        return metrics

    def snapshot(self) -> str:
        return self.trainer.save_state(name=f"nenyax-snap-{self.step}").result().path

    def restore(self, path: str) -> None:
        self.trainer.load_state_with_optimizer(path).result()
        self.sampler = self.trainer.save_weights_and_get_sampling_client(
            name=f"nenyax-restored-{self.step}"
        )
