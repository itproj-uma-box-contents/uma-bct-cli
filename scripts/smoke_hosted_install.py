#!/usr/bin/env python3
"""Check the installed hosted extra using the real SDK and synthetic replies."""

from __future__ import annotations

import os
from unittest.mock import patch


def main() -> None:
    # Import the SDK's bundled metadata instead of fetching its remote price
    # map. All completions below are mocked; no provider receives a request.
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    import litellm

    from uma_bct.adapters.hosted_client import HostedVisionModel
    from uma_bct.model_config import HostedModelConfig

    response = {"choices": [{"finish_reason": "stop", "message": {"content": '{"dates":[]}'}}]}
    schema = {"type": "object", "properties": {"dates": {"type": "array", "items": {"type": "string"}}},
              "required": ["dates"]}
    for provider in ("openai", "anthropic", "gemini"):
        client = HostedVisionModel(HostedModelConfig(f"{provider}/synthetic-model", "synthetic-key"))
        client.preflight()
        with patch.object(litellm, "completion", return_value=response) as completion:
            result = client.query(None, "Synthetic installation check.", schema)
        if result.value != {"dates": []} or completion.call_count != 1:
            raise RuntimeError(f"Installed hosted adapter failed its {provider} smoke test.")
    print("Installed hosted adapters passed with synthetic replies.")


if __name__ == "__main__":
    main()
