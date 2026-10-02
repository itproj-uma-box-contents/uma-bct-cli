"""Read the optional hosted-model selection without exposing credentials."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

API_KEY_ENVIRONMENT_VARIABLE = "UMA_BCT_API_KEY"
API_MODEL_ENVIRONMENT_VARIABLE = "UMA_BCT_API_MODEL"


@dataclass(frozen=True)
class HostedModelConfig:
    model: str
    api_key: str = field(repr=False)

    @property
    def provider(self) -> str:
        return self.model.split("/", 1)[0]


def selected_model_name(arguments: Any) -> str:
    """Name the model actually used by this run without retaining its key."""

    hosted = getattr(arguments, "hosted_model", None)
    return hosted.model if hosted is not None else arguments.model


def selected_model_metadata(arguments: Any) -> dict[str, str]:
    """Public run identity; credentials never enter manifests or checkpoints."""

    hosted = getattr(arguments, "hosted_model", None)
    if hosted is not None:
        return {"model": hosted.model, "backend": "hosted", "provider": hosted.provider}
    return {"model": arguments.model}


def hosted_model_from_environment(environ: Mapping[str, str] | None = None) -> HostedModelConfig | None:
    """An empty key selects local Qwen; a key requires a provider-qualified model."""

    values = os.environ if environ is None else environ
    api_key = values.get(API_KEY_ENVIRONMENT_VARIABLE, "").strip()
    if not api_key:
        return None

    model = values.get(API_MODEL_ENVIRONMENT_VARIABLE, "").strip()
    if not model:
        raise ValueError(f"{API_MODEL_ENVIRONMENT_VARIABLE} is required when {API_KEY_ENVIRONMENT_VARIABLE} is set.")
    if any(character.isspace() for character in model) or any(not part for part in model.split("/")) or "/" not in model:
        raise ValueError(f"{API_MODEL_ENVIRONMENT_VARIABLE} must use provider/model format.")
    return HostedModelConfig(model=model, api_key=api_key)
