"""Optional Azure OpenAI mode (USE_AZURE_OPENAI=true).

Sends exactly the same messages that were counted offline and returns the
real usage reported by the API, so the estimate can be validated.
Credentials come from environment variables only.
"""
from __future__ import annotations

import json

import config

REQUIRED_SETTINGS = [
    "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT",
]


def create_client():
    missing = [name for name in REQUIRED_SETTINGS if not getattr(config, name)]
    if missing:
        raise RuntimeError(
            "USE_AZURE_OPENAI=true but these settings are empty: " + ", ".join(missing)
        )
    try:
        from openai import AzureOpenAI
    except ImportError as exc:
        raise RuntimeError("Azure mode needs the openai package: pip install openai") from exc
    return AzureOpenAI(
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
    )


def assess(client, messages: list[dict], deployment: str | None = None) -> dict:
    """Call the deployment once for one hit and capture actual token usage."""
    kwargs = {
        "model": deployment or config.AZURE_OPENAI_DEPLOYMENT,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    if config.AZURE_OPENAI_TEMPERATURE != "":
        kwargs["temperature"] = float(config.AZURE_OPENAI_TEMPERATURE)

    response = client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content or ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    valid = isinstance(parsed, dict) and parsed.get("recommendation") in config.AI_RECOMMENDATIONS
    return {
        "response_text": text,
        "response": parsed,
        "response_valid": valid,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
