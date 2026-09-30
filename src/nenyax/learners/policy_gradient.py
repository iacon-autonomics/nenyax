"""Policy-gradient RL from verifiable rewards, on a local open-weight model (PyTorch).

One learner, composable parts, named presets for the published recipes::

    PolicyGradientLearner.preset("dapo", "Qwen/Qwen2.5-0.5B-Instruct")
    PolicyGradientLearner(model, advantage="rloo", loss="ppo", epochs=2, clip=(0.2, 0.28))

========== ========== ======== ============ =======================================
preset     advantage  loss     aggregation  notes
========== ========== ======== ============ =======================================
grpo       grpo       ppo      sequence     DeepSeekMath GRPO (group mean/std)
dr_grpo    dr_grpo    ppo      token        Dr. GRPO: no std, no length bias
dapo       grpo       ppo      token        DAPO: clip-higher (0.2, 0.28)
rloo       rloo       reinforce sequence    leave-one-out baseline
reinforce  reinforce  reinforce sequence    REINFORCE with a batch-mean baseline
reinforce_pp batch_norm ppo    token        REINFORCE++-style batch normalization
gspo       grpo       gspo     sequence     GSPO: sequence-level importance ratio
========== ========== ======== ============ =======================================

The learner is also the policy: it generates with the current weights and keeps the exact token
ids it sampled, so updates use what the model actually produced. Sampling uses the model's full
distribution (no top-k/top-p), so the loss is on-policy. ``kl`` > 0 adds a penalty against a
frozen copy of the starting weights.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any, Literal

from ..types import Message, Trajectory
from .advantages import compute
from .base import has_signal

_IDS = "nenyax_ids"

PRESETS: dict[str, dict[str, Any]] = {
    "grpo": {"advantage": "grpo", "loss": "ppo", "aggregate": "sequence"},
    "dr_grpo": {"advantage": "dr_grpo", "loss": "ppo", "aggregate": "token"},
    "dapo": {"advantage": "grpo", "loss": "ppo", "aggregate": "token", "clip": (0.2, 0.28)},
    "rloo": {"advantage": "rloo", "loss": "reinforce", "aggregate": "sequence"},
    "reinforce": {"advantage": "reinforce", "loss": "reinforce", "aggregate": "sequence"},
    "reinforce_pp": {"advantage": "batch_norm", "loss": "ppo", "aggregate": "token"},
    "gspo": {"advantage": "grpo", "loss": "gspo", "aggregate": "sequence", "clip": (3e-4, 4e-4)},
}


class _Chat:
    """ChatModel backed by the learner's current weights; keeps sampled token ids."""

    def __init__(self, learner: HFLearnerBase) -> None:
        self.learner, self.model = learner, learner.name

    def complete(self, messages: Sequence[Message], **params: Any) -> Message:
        import torch

        lr = self.learner
        chat = [
            {"role": m.role, "content": m.content or ""}
            for m in messages
            if isinstance(m.content, str) or m.content is None
        ]
        enc = lr.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        prompt_ids = enc["input_ids"].to(lr.device)
        with torch.no_grad():
            out = lr.model.generate(
                prompt_ids,
                attention_mask=enc["attention_mask"].to(lr.device),
                do_sample=lr.temperature > 0,
                temperature=max(lr.temperature, 1e-5),
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.0,
                max_new_tokens=int(params.get("max_tokens") or lr.max_new_tokens),
                pad_token_id=lr.tokenizer.pad_token_id or lr.tokenizer.eos_token_id,
            )
        completion = out[0, prompt_ids.shape[1] :].tolist()
        return Message.model_validate(
            {
                "role": "assistant",
                "content": lr.tokenizer.decode(completion, skip_special_tokens=True),
                _IDS: {"prompt": prompt_ids[0].tolist(), "completion": completion},
            }
        )


class HFLearnerBase:
    """Shared plumbing for learners that own a Hugging Face causal LM."""

    def __init__(
        self,
        model: str,
        *,
        lr: float,
        max_new_tokens: int,
        temperature: float,
        max_grad_norm: float,
        device: str | None,
        dtype: str,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, dtype)).to(
            self.device
        )
        self.model.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.max_new_tokens, self.temperature = max_new_tokens, temperature
        self.max_grad_norm = max_grad_norm

    def policy(self) -> _Chat:
        return _Chat(self)

    @staticmethod
    def samples(traj: Trajectory) -> list[tuple[list[int], list[int]]]:
        """(prompt ids, completion ids) for every model call this learner produced."""
        out = []
        for call in traj.model_calls:
            message = ((call.response or {}).get("choices") or [{}])[0].get("message") or {}
            ids = message.get(_IDS)
            if ids and ids["completion"]:
                out.append((ids["prompt"], ids["completion"]))
        return out

    def token_logprobs(self, model: Any, prompt: list[int], completion: list[int]) -> Any:
        import torch

        ids = torch.tensor([prompt + completion], device=self.device)
        logits = model(ids).logits[0, len(prompt) - 1 : -1].float()
        target = torch.tensor(completion, device=self.device)
        return torch.log_softmax(logits, dim=-1).gather(1, target[:, None]).squeeze(1)

    def step(self) -> None:
        import torch

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def snapshot(self) -> dict[str, Any]:
        return {
            "model": {
                k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()
            },
            "optim": copy.deepcopy(self.optimizer.state_dict()),
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optim"])

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


class PolicyGradientLearner(HFLearnerBase):
    def __init__(
        self,
        model: str,
        *,
        advantage: str = "grpo",
        loss: Literal["reinforce", "ppo", "gspo"] = "ppo",
        aggregate: Literal["token", "sequence"] = "sequence",
        clip: tuple[float, float] = (0.2, 0.2),
        epochs: int = 1,
        kl: float = 0.0,
        lr: float = 1e-6,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        max_grad_norm: float = 1.0,
        device: str | None = None,
        dtype: str = "float32",
    ) -> None:
        """
        Args:
            advantage: ``grpo | dr_grpo | rloo | reinforce | batch_norm``.
            loss: ``reinforce`` (-A·logπ), ``ppo`` (token-level clipped ratio) or ``gspo``
                (sequence-level clipped ratio).
            aggregate: average the loss per ``token`` across the batch, or per ``sequence``.
            clip: (ε_low, ε_high) for the ratio; ε_high > ε_low is DAPO's clip-higher.
            epochs: optimisation passes over each batch (ratios differ from 1 after the first).
            kl: weight of a KL penalty against the initial weights (0 disables it).
        """
        super().__init__(
            model,
            lr=lr,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            max_grad_norm=max_grad_norm,
            device=device,
            dtype=dtype,
        )
        if loss not in ("reinforce", "ppo", "gspo"):
            raise ValueError(f"loss must be reinforce | ppo | gspo, got {loss!r}")
        self.advantage, self.loss, self.aggregate = advantage, loss, aggregate
        self.clip, self.epochs, self.kl = clip, epochs, kl
        self.reference = None
        if kl > 0:
            self.reference = copy.deepcopy(self.model).eval()
            for p in self.reference.parameters():
                p.requires_grad_(False)

    @classmethod
    def preset(cls, name: str, model: str, **overrides: Any) -> PolicyGradientLearner:
        try:
            recipe = dict(PRESETS[name])
        except KeyError:
            raise ValueError(f"unknown preset {name!r}; have {sorted(PRESETS)}") from None
        return cls(model, **{**recipe, **overrides})

    def _objective(self, logp: Any, old: Any, adv: float, ref: Any) -> Any:
        """Per-token loss terms (to be aggregated) for one sampled completion."""
        import torch

        low, high = self.clip
        if self.loss == "reinforce":
            terms = -adv * logp
        elif self.loss == "ppo":
            ratio = torch.exp(logp - old)
            terms = -torch.minimum(ratio * adv, torch.clamp(ratio, 1 - low, 1 + high) * adv)
        else:  # gspo: one length-normalised ratio for the whole sequence
            ratio = torch.exp((logp - old).mean())
            seq = -torch.minimum(ratio * adv, torch.clamp(ratio, 1 - low, 1 + high) * adv)
            terms = seq.expand_as(logp)
        if ref is not None:  # k3 estimator of KL(π || π_ref)
            diff = ref - logp
            terms = terms + self.kl * (torch.exp(diff) - diff - 1)
        return terms

    def update(self, groups: list[list[Trajectory]]) -> dict[str, float]:
        import torch

        batch = [
            (adv, sample)
            for group, advs in zip(groups, compute(self.advantage, groups), strict=True)
            for traj, adv in zip(group, advs, strict=True)
            if adv != 0.0
            for sample in self.samples(traj)
        ]
        metrics = {"signal": has_signal(groups), "samples": float(len(batch)), "loss": 0.0}
        if not batch:
            return metrics
        with torch.no_grad():  # the sampling policy's log-probs (π_old) and the reference's
            old = [self.token_logprobs(self.model, p, c) for _, (p, c) in batch]
            ref = (
                [self.token_logprobs(self.reference, p, c) for _, (p, c) in batch]
                if self.reference is not None
                else [None] * len(batch)
            )
        n_tokens = sum(len(c) for _, (_, c) in batch)
        self.model.train()
        total, clipped = 0.0, 0
        for _ in range(self.epochs):
            for (adv, (prompt, completion)), lp_old, lp_ref in zip(batch, old, ref, strict=True):
                logp = self.token_logprobs(self.model, prompt, completion)
                terms = self._objective(logp, lp_old, adv, lp_ref)
                if self.aggregate == "token":
                    loss = terms.sum() / n_tokens
                else:
                    loss = terms.mean() / len(batch)
                loss.backward()
                total += loss.item()
                ratio = torch.exp(logp.detach() - lp_old)
                clipped += int(((ratio < 1 - self.clip[0]) | (ratio > 1 + self.clip[1])).sum())
            self.step()
        self.model.eval()
        metrics.update(loss=total / self.epochs, clip_frac=clipped / (n_tokens * self.epochs))
        return metrics


class RejectionSamplingLearner(HFLearnerBase):
    """Rejection-sampling fine-tuning (RFT / STaR / expert iteration).

    Keep only rollouts scoring at least ``min_score`` (optionally the best one per group) and
    run supervised cross-entropy on them. Simple, stable, and a strong baseline to beat.
    """

    def __init__(
        self,
        model: str,
        *,
        min_score: float = 1.0,
        best_of_group: bool = True,
        lr: float = 5e-6,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        max_grad_norm: float = 1.0,
        device: str | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__(
            model,
            lr=lr,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            max_grad_norm=max_grad_norm,
            device=device,
            dtype=dtype,
        )
        self.min_score, self.best_of_group = min_score, best_of_group

    def update(self, groups: list[list[Trajectory]]) -> dict[str, float]:
        keep = []
        for group in groups:
            good = [t for t in group if (t.score or 0.0) >= self.min_score]
            if self.best_of_group and good:
                good = [max(good, key=lambda t: t.score or 0.0)]
            keep += [s for t in good for s in self.samples(t)]
        metrics = {"kept": float(len(keep)), "loss": 0.0}
        if not keep:
            return metrics
        self.model.train()
        total = 0.0
        for prompt, completion in keep:
            loss = -self.token_logprobs(self.model, prompt, completion).mean() / len(keep)
            loss.backward()
            total += loss.item()
        self.step()
        self.model.eval()
        metrics["loss"] = total
        return metrics


class DPOLearner(HFLearnerBase):
    """Direct preference optimisation on pairs built from each group (best vs. worst rollout).

    The reference is a frozen copy of the starting weights. ``beta`` scales the implicit reward.
    """

    def __init__(
        self,
        model: str,
        *,
        beta: float = 0.1,
        lr: float = 1e-6,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        max_grad_norm: float = 1.0,
        device: str | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__(
            model,
            lr=lr,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            max_grad_norm=max_grad_norm,
            device=device,
            dtype=dtype,
        )
        self.beta = beta
        self.reference = copy.deepcopy(self.model).eval()
        for p in self.reference.parameters():
            p.requires_grad_(False)

    def update(self, groups: list[list[Trajectory]]) -> dict[str, float]:
        import torch
        import torch.nn.functional as F

        pairs = []
        for group in groups:
            best = max(group, key=lambda t: t.score or 0.0)
            worst = min(group, key=lambda t: t.score or 0.0)
            if (best.score or 0.0) > (worst.score or 0.0):
                chosen, rejected = self.samples(best), self.samples(worst)
                if chosen and rejected:
                    pairs.append((chosen[-1], rejected[-1]))
        metrics = {"pairs": float(len(pairs)), "loss": 0.0, "accuracy": 0.0}
        if not pairs:
            return metrics
        self.model.train()
        total, correct = 0.0, 0
        for (cp, cc), (rp, rc) in pairs:
            pi_c = self.token_logprobs(self.model, cp, cc).sum()
            pi_r = self.token_logprobs(self.model, rp, rc).sum()
            with torch.no_grad():
                ref_c = self.token_logprobs(self.reference, cp, cc).sum()
                ref_r = self.token_logprobs(self.reference, rp, rc).sum()
            margin = self.beta * ((pi_c - ref_c) - (pi_r - ref_r))
            loss = -F.logsigmoid(margin) / len(pairs)
            loss.backward()
            total += loss.item()
            correct += int(margin.item() > 0)
        self.step()
        self.model.eval()
        metrics.update(loss=total, accuracy=correct / len(pairs))
        return metrics
