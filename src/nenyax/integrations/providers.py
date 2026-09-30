"""Model providers: every hosted or local server that speaks OpenAI chat completions.

    policy = nenyax.integrations.endpoint("together", "Qwen/Qwen3-8B")   # reads TOGETHER_API_KEY
    policy = nenyax.integrations.endpoint("vllm", "Qwen/Qwen3-8B")       # localhost:8000

Adding a provider is one line in ``PROVIDERS``, or a ``nenyax.providers`` entry point returning a
:class:`Provider`.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points

from ..environment import NenyaxError
from ..policy import Endpoint
from .base import Integration


@dataclass(frozen=True)
class ProviderSpec:
    base_url: str
    key_var: str | None
    """Environment variable holding the API key; ``None`` for local servers."""
    token_ids: bool = False
    """Returns token ids / logprobs usable for training (vLLM, SGLang)."""
    note: str = ""


PROVIDERS: dict[str, ProviderSpec] = {
    # hosted
    "openai": ProviderSpec("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "anthropic": ProviderSpec(
        "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY", note="OpenAI-compatible endpoint"
    ),
    "gemini": ProviderSpec(
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "GEMINI_API_KEY",
        note="OpenAI-compatible endpoint",
    ),
    "mistral": ProviderSpec("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "openrouter": ProviderSpec("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "together": ProviderSpec("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "fireworks": ProviderSpec("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "groq": ProviderSpec("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "cerebras": ProviderSpec("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
    "deepinfra": ProviderSpec("https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY"),
    "huggingface": ProviderSpec(
        "https://router.huggingface.co/v1", "HF_TOKEN", note="HF Inference Providers router"
    ),
    "prime": ProviderSpec(
        "https://api.pinference.ai/api/v1", "PRIME_API_KEY", note="Prime Intellect inference"
    ),
    # local (no key)
    "vllm": ProviderSpec("http://127.0.0.1:8000/v1", None, token_ids=True),
    "sglang": ProviderSpec("http://127.0.0.1:30000/v1", None, token_ids=True),
    "ollama": ProviderSpec("http://127.0.0.1:11434/v1", None),
    "lmstudio": ProviderSpec("http://127.0.0.1:1234/v1", None),
    "mlx": ProviderSpec("http://127.0.0.1:8080/v1", None, note="mlx_lm.server"),
    "llamacpp": ProviderSpec("http://127.0.0.1:8080/v1", None, note="llama-server"),
}


class Provider(Integration):
    kind = "provider"

    def __init__(self, name: str, spec: ProviderSpec, **credentials: str) -> None:
        super().__init__(**credentials)
        self.name, self.spec = name, spec
        self.credentials = (spec.key_var,) if spec.key_var else ()
        self.summary = spec.note or spec.base_url

    def endpoint(self, model: str, *, base_url: str | None = None, **params) -> Endpoint:
        missing = self.missing()
        if missing:
            raise NenyaxError(f"provider {self.name!r} is not ready: {'; '.join(missing)}")
        key = self.credential(self.spec.key_var) if self.spec.key_var else "EMPTY"
        return Endpoint(base_url or self.spec.base_url, model, api_key=key, default_params=params)


def providers() -> dict[str, Provider]:
    found = {name: Provider(name, spec) for name, spec in PROVIDERS.items()}
    for ep in entry_points(group="nenyax.providers"):
        provider = ep.load()()
        found[provider.name] = provider
    return found


def endpoint(
    provider: str, model: str, *, base_url: str | None = None, api_key: str | None = None, **params
) -> Endpoint:
    """An :class:`Endpoint` for ``model`` at ``provider``; the key comes from its env var."""
    try:
        p = providers()[provider]
    except KeyError:
        raise NenyaxError(f"unknown provider {provider!r}; have {sorted(providers())}") from None
    if api_key and p.spec.key_var:
        p = Provider(p.name, p.spec, **{p.spec.key_var: api_key})
    return p.endpoint(model, base_url=base_url, **params)
