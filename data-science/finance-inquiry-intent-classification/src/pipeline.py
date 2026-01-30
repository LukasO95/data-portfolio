import json

from jsonschema import ValidationError

from .pii_masking import mask_pii
from .classifier import classify_finance_inquiry
from .normalization import normalize_llm_output
from .validation import validate_output
from .routing import derive_routing
from .crm import build_crm_payload
from .versioning import CLASSIFIER_VERSION
from .config import DEFAULT_THRESHOLD, DEFAULT_MODEL


def _is_empty_text(text):
    """True if text is None, empty, or only whitespace."""
    return not text or not text.strip()


def _fallback_payload(raw_text, reason_code, error):
    """
    Safe fallback payload: route to manual clearing.
    Keep fields CRM-compatible and explicit.
    """
    return {
        "message": raw_text,
        "label": "unclear",
        "confidence": 0.0,
        "handoff": "clearing",
        "product_area": "none",
        "service_area": "none",
        "reason_codes": [reason_code],
        "needs_human": True,
        "classifier_version": CLASSIFIER_VERSION,
        "error": str(error),
    }


def classify_and_prepare_crm(raw_text, threshold=DEFAULT_THRESHOLD, model=DEFAULT_MODEL):
    """
    End-to-end pipeline:
    - check for empty text
    - mask PII for LLM input
    - classify via LLM (mock or OpenAI provider)
    - normalize the LLM output (defensive)
    - validate against schema (contract)
    - derive deterministic routing policy (handoff vs. needs_human)
    - build CRM-ready payload
    """

    if _is_empty_text(raw_text):
        return {
            "message": raw_text,
            "label": "unclear",
            "confidence": 0.0,
            "handoff": "clearing",
            "product_area": "none",
            "service_area": "none",
            "reason_codes": ["insufficient_information"],
            "needs_human": True,
            "classifier_version": CLASSIFIER_VERSION,
        }

    try:
        masked = mask_pii(raw_text)

        # 1) LLM classification (may raise provider/JSON errors)
        classification = classify_finance_inquiry(masked_text=masked, model=model)

        # 2) Defensive normalization (avoid schema breaks from minor slips)
        classification = normalize_llm_output(classification)

        # 3) Contract enforcement (schema validation)
        validate_output(classification)

        # 4) Routing policy (operational fields, not part of LLM schema)
        routing = derive_routing(classification, threshold=threshold)

        # 5) CRM payload (keep original raw text for CRM)
        return build_crm_payload(raw_text, classification, routing)

    except ValidationError as e:
        return _fallback_payload(raw_text, "schema_validation_failed", e)

    except json.JSONDecodeError as e:
        return _fallback_payload(raw_text, "invalid_json_from_llm", e)

    except Exception as e:
        # Covers rate limits, network issues, missing API key, provider failures, etc.
        return _fallback_payload(raw_text, "classifier_runtime_error", e)
