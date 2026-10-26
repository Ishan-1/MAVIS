"""
core/decision package
Provides Decision Model abstractions and factory for MAVIS.
Decouples structured decision-making (Choice, Score, Noul) from conversational text generation.
"""
from __future__ import annotations

import os
from typing import Any

from core.config import cfg
from core.decision.base import BaseDecisionModel, DecisionResult
from core.decision.typesafe import TypeSafeDecisionModel
from core.llm.base import BaseLLMClient

_MODEL_CACHE: BaseDecisionModel | None = None


def get_decision_model(
    provider: str | None = None,
    fallback_client: BaseLLMClient | None = None,
    force_refresh: bool = False,
    **kwargs: Any,
) -> BaseDecisionModel:
    """
    Factory returning the configured BaseDecisionModel instance.
    Defaults to TypeSafeDecisionModel (with automatic LLM fallback built-in).
    """
    global _MODEL_CACHE

    ts_cfg = cfg.get("typesafe", default={})
    if not isinstance(ts_cfg, dict):
        ts_cfg = {}

    target_provider = (provider or ts_cfg.get("provider", "typesafe")).lower()
    api_key = kwargs.get("api_key") or os.getenv("TYPESAFE_API_KEY") or ts_cfg.get("api_key", "")
    base_url = kwargs.get("base_url") or ts_cfg.get("base_url", "https://api.typesafe.ai/v1")
    timeout = float(kwargs.get("timeout") or ts_cfg.get("timeout", 5.0))

    # If caller specifies a specific fallback client, instantiate directly for caller
    if fallback_client is not None:
        if target_provider == "typesafe":
            return TypeSafeDecisionModel(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                fallback_client=fallback_client,
            )
        return BaseDecisionModel(fallback_client=fallback_client)

    if _MODEL_CACHE is not None and not force_refresh:
        return _MODEL_CACHE

    if target_provider == "typesafe":
        model = TypeSafeDecisionModel(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            fallback_client=None,
        )
    else:
        model = BaseDecisionModel(fallback_client=None)

    _MODEL_CACHE = model
    return model


__all__ = [
    "BaseDecisionModel",
    "DecisionResult",
    "TypeSafeDecisionModel",
    "get_decision_model",
]
