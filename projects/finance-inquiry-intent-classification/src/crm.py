from .versioning import CLASSIFIER_VERSION


def build_crm_payload(raw_text, classification, routing):
    return {
        "message": raw_text,
        "label": classification["label"],
        "confidence": float(classification["confidence"]),
        "product_area": classification["product_area"],
        "service_area": classification["service_area"],
        "reason_codes": classification.get("reason_codes", []),
        "handoff": routing["handoff"],
        "needs_human": bool(routing["needs_human"]),
        "classifier_version": CLASSIFIER_VERSION,
    }
