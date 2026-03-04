from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI

LOGGER = logging.getLogger(__name__)


class LLMClient:
    """Small OpenAI wrapper with graceful fallback when key is unavailable."""

    def __init__(self, api_key: str | None, model: str) -> None:
        self.enabled = bool(api_key)
        self.model = model
        self.client = OpenAI(api_key=api_key) if self.enabled else None

    def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str | None:
        if not self.enabled or self.client is None:
            return None

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> dict[str, Any] | None:
        raw = self.complete_text(system_prompt, user_prompt, temperature)
        if not raw:
            return None

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                LOGGER.warning("LLM JSON parsing failed: no JSON object detected")
                return None

            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                LOGGER.warning("LLM JSON parsing failed after extraction")
                return None
