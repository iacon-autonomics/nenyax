"""Integrations with the platforms around RL environments.

One interface per kind, bring-your-own-key for hosted services, entry points for everything:

    kind       entry-point group     built in
    provider   nenyax.providers      OpenAI, Anthropic, Gemini, OpenRouter, Together, ... , vLLM
    hub        nenyax.hubs           Hugging Face (OpenEnv Spaces), Prime Environments Hub, Harbor
    sandbox    nenyax.sandboxes      process, docker (see nenyax.sandbox)
    telemetry  nenyax.telemetry      OpenTelemetry, MLflow, Weights & Biases
    export     (functions)           JSONL, Parquet, HF datasets / Hub, in raw, SFT or DPO shape
"""

from . import export, telemetry
from .base import Integration
from .hubs import Hub, Listing, fetch, hubs, search
from .providers import PROVIDERS, Provider, ProviderSpec, endpoint, providers

__all__ = [
    "PROVIDERS",
    "Hub",
    "Integration",
    "Listing",
    "Provider",
    "ProviderSpec",
    "endpoint",
    "export",
    "fetch",
    "hubs",
    "providers",
    "search",
    "telemetry",
]
