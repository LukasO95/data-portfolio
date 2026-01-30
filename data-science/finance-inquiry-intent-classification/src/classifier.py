from .config import DEFAULT_MODEL
from .llm_provider import get_provider


def classify_finance_inquiry(masked_text, model=DEFAULT_MODEL):
    provider = get_provider()
    return provider.classify(masked_text=masked_text, model=model)
