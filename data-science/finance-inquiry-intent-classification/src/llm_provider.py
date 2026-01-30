import os
import json
from typing import Protocol

from .prompts import load_classifier_prompt
from .utils.rate_limit import SimpleRateLimiter

# Rate limiter for OpenAI (e.g., 10 calls per minute)
_openai_rate_limiter = SimpleRateLimiter(max_calls=10, per_seconds=60)

# Define method signature for LLM providers (standard interface)
class LLMProvider(Protocol):
    def classify(self, masked_text, model): ...


def _base_result():
    """Create fallback result for invalid LLM outputs."""
    return {
        "label": "unclear",
        "confidence": 0.0,
        "product_area": "none",
        "service_area": "none",
        "reason_codes": ["insufficient_information"],
        "schema_version": "v1",
    }


def _strip_code_fences(s):
    """
    Remove wrappers (```) if the model ever returns fenced JSON
    despite prompt instructions.
    """
    s = (s or "").strip()
    if not s.startswith("```"):
        return s

    # Remove fence lines if present
    first_nl = s.find("\n")
    if first_nl != -1:  #
        s = s[first_nl + 1 :]

    s = s.strip()
    if s.endswith("```"):
        s = s[:-3]

    return s.strip()


class MockLLMProvider:
    """
    Minimal demo mock. Used when OpenAI access is not available.
    Provides a technically valid but meaningless result.
    """

    def classify(self, masked_text, model="mock"):
        t = masked_text.lower()

        # Decide roughly between product vs. service inquiry
        if "kredit" in t or "darlehen" in t:
            return {
                "label": "product",
                "confidence": 0.9,
                "product_area": "mortgage",
                "service_area": "none",
                "reason_codes": ["rate_or_offer_request"],
                "schema_version": "v1",
            }

        if "login" in t or "zugang" in t:
            return {
                "label": "service",
                "confidence": 0.9,
                "product_area": "none",
                "service_area": "online_banking",
                "reason_codes": ["online_access_issue"],
                "schema_version": "v1",
            }

        # Default: unclear inquiry
        return {
            "label": "unclear",
            "confidence": 0.5,
            "product_area": "none",
            "service_area": "none",
            "reason_codes": ["insufficient_information"],
            "schema_version": "v1",
        }


class OpenAILLMProvider:
    def __init__(self):
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        self.client = OpenAI(api_key=api_key)

    def classify(self, masked_text, model):
        instructions = load_classifier_prompt()

        _openai_rate_limiter.acquire()

        resp = self.client.responses.create(
            model=model,
            instructions=instructions,
            input=masked_text,
        )

        text = _strip_code_fences(resp.output_text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return _base_result()

        # trust prompt/schema contract only if we got a dict
        if not isinstance(data, dict):
            return _base_result()

        return data


def get_provider():
    provider = os.getenv("LLM_PROVIDER", "mock").lower()
    if provider == "openai":
        return OpenAILLMProvider()
    return MockLLMProvider()