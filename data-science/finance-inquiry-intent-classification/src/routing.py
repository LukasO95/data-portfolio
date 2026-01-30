# Implements routing logic based on classification output

from .config import DEFAULT_THRESHOLD

VALID_LABELS = {"product", "service", "unclear"}


def derive_routing(classification, threshold=DEFAULT_THRESHOLD):
    label = classification.get("label")
    conf = float(classification.get("confidence", 0.0))

    if label not in VALID_LABELS:
        label = "unclear"
        conf = 0.0

    needs_human = (label == "unclear") or (conf < threshold)

    if needs_human:
        handoff = "clearing"
    else:
        handoff = "sales" if label == "product" else "support"

    return {"handoff": handoff, "needs_human": bool(needs_human)}