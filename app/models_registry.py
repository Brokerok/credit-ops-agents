"""One place that knows how to build a model from a spec string and what it costs.

Spec format: "<provider>:<model_name>", e.g. "anthropic:claude-sonnet-4-5", "openai:gpt-4.1-mini",
"ollama:llama3.1:8b", "test:test". The rest of the code never imports provider SDKs directly, so swapping
models is a config change, and the eval harness can compare several in one run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai import PromptedOutput
from pydantic_ai.models import Model

from app.config import settings

# USD per 1M tokens (input, output). Used for the cost column in evals and run records; keep it current.
PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
}


@dataclass(frozen=True)
class ModelProfile:
    spec: str
    provider: str
    name: str
    model: Model
    price_in: float  # USD per 1M input tokens
    price_out: float
    native_tools: bool  # can we trust the model to drive tool calls?

    def output_type(self, schema: type[Any]) -> Any:
        """Local models are most reliable with prompted JSON; hosted models get native structured output."""
        return PromptedOutput(schema) if self.provider == "ollama" else schema

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return round((input_tokens * self.price_in + output_tokens * self.price_out) / 1_000_000, 6)


def build_model(spec: str | None = None) -> ModelProfile:
    spec = spec or settings.llm_model
    provider, _, name = spec.partition(":")
    if provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel

        model: Model = AnthropicModel(name)
        tools = True
    elif provider == "openai":
        from pydantic_ai.models.openai import OpenAIChatModel

        model = OpenAIChatModel(name)
        tools = True
    elif provider == "ollama":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.ollama import OllamaProvider

        model = OpenAIChatModel(name, provider=OllamaProvider(base_url=settings.ollama_base_url))
        tools = False  # 8B-class local models: keep tool use deterministic, in code (see orchestrator)
    elif provider == "test":
        from pydantic_ai.models.test import TestModel

        model = TestModel()
        tools = False
    else:
        raise ValueError(f"unknown model provider in spec {spec!r}")
    price_in, price_out = PRICES.get(name, (0.0, 0.0))
    return ModelProfile(spec, provider, name, model, price_in, price_out, tools)
