import json
from pathlib import Path
import os

from jsonschema import ValidationError

from .pii_masking import mask_pii
from .classifier import classify_finance_inquiry
from .normalization import normalize_llm_output
from .validation import validate_output
from .routing import derive_routing
from .config import DEFAULT_THRESHOLD, DEFAULT_MODEL

# Ensure the evaluation runs with the OpenAI LLM provider
assert os.getenv("LLM_PROVIDER") == "openai", "Eval must run with LLM_PROVIDER=openai"

# Define the path to the evaluation samples JSON file
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "eval_samples.json"


def load_eval_samples():
    try:
        text = DATA_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Eval samples not found: {DATA_PATH}") from e

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {DATA_PATH}: {e}") from e

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a list of samples in {DATA_PATH}, got {type(data).__name__}"
        )

    return data


def evaluate(samples, threshold=DEFAULT_THRESHOLD, model=DEFAULT_MODEL):
    rows = []

    for s in samples:
        raw_text = s["text"]
        expected = s["expected"]

        try:
            masked = mask_pii(raw_text)

            classification = classify_finance_inquiry(masked_text=masked, model=model)
            classification = normalize_llm_output(classification)
            validate_output(classification)

            routing = derive_routing(classification, threshold=threshold)

            rows.append(
                {
                    "expected": expected,
                    "predicted": classification["label"],
                    "confidence": float(classification["confidence"]),
                    "needs_human": bool(routing["needs_human"]),
                    "handoff": routing["handoff"],
                }
            )

        except (json.JSONDecodeError, ValidationError, Exception) as e:
            # In evaluation failures are treated as "unclear" (conservative)
            rows.append(
                {
                    "expected": expected,
                    "predicted": "unclear",
                    "confidence": 0.0,
                    "needs_human": True,
                    "handoff": "clearing",
                    "error": str(e),
                }
            )

    total = len(rows)
    overall_acc = (
        (sum(r["expected"] == r["predicted"] for r in rows) / total) if total else 0.0
    )

    ps = [r for r in rows if r["expected"] in ("product", "service")]
    ps_acc = (sum(r["expected"] == r["predicted"] for r in ps) / len(ps)) if ps else 0.0

    #  accuracy on auto-routable cases only (needs_human=False)
    auto = [r for r in rows if r.get("needs_human") is False]
    auto_acc = (
        (sum(r["expected"] == r["predicted"] for r in auto) / len(auto))
        if auto
        else 0.0
    )

    unclear_rate = (
        (sum(r["predicted"] == "unclear" for r in rows) / total) if total else 0.0
    )

    return {
        "overall_accuracy": overall_acc,
        "product_service_accuracy": ps_acc,
        "auto_case_accuracy": auto_acc,
        "unclear_rate": unclear_rate,
        "n_samples": total,
        "n_auto_cases": len(auto),
        "details": rows,
    }


if __name__ == "__main__":
    samples = load_eval_samples()
    report = evaluate(samples, threshold=0.75)
    print(json.dumps(report, indent=2, ensure_ascii=False))
