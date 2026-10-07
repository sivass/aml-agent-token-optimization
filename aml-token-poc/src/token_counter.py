"""Offline token estimation. No Azure key required.

Uses tiktoken when its vocabulary can be loaded. tiktoken downloads that
vocabulary on first use; on a locked-down network that download can fail,
so the counter falls back to a characters-per-token estimate and says so.

Offline tokenization is an estimate. Exact production token usage must be
validated using the actual Azure OpenAI deployment.
"""
from __future__ import annotations

import math

import config

# Chat format overhead for GPT-4o-family models (OpenAI cookbook):
# 3 tokens per message, plus 3 to prime the assistant reply.
TOKENS_PER_MESSAGE = 3
TOKENS_REPLY_PRIMING = 3


class TokenCounter:
    def __init__(self, model: str | None = None, allow_tiktoken: bool = True):
        self.model = model or config.TOKENIZER_MODEL
        self._encoding = None
        self.method = f"heuristic (~{config.HEURISTIC_CHARS_PER_TOKEN} chars/token)"
        if allow_tiktoken:
            self._load_tiktoken()

    def _load_tiktoken(self) -> None:
        try:
            import tiktoken
            try:
                self._encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:  # model name unknown to this tiktoken version
                self._encoding = tiktoken.get_encoding("o200k_base")
            self.method = f"tiktoken ({self._encoding.name})"
        except Exception:  # not installed, or vocabulary download blocked
            self._encoding = None

    @property
    def is_exact_tokenizer(self) -> bool:
        return self._encoding is not None

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        return math.ceil(len(text) / config.HEURISTIC_CHARS_PER_TOKEN)

    def count_messages(self, messages: list[dict]) -> int:
        """Input tokens for a chat request, including message framing."""
        total = TOKENS_REPLY_PRIMING
        for message in messages:
            total += TOKENS_PER_MESSAGE
            total += self.count_text(message["role"]) + self.count_text(message["content"])
        return total
