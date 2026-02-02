# Implements normalization and cross-field consistency for LLM output

PRODUCT_AREAS = {"mortgage", "savings", "home_savings", "other", "none"}
SERVICE_AREAS = {"existing_contract", "online_banking", "documents", "payments", "complaints", "other", "none"}

def normalize_llm_output(d):
    d = dict(d)
    label = d.get("label")

    # Normalize product_area
    if label == "product":
        if d.get("product_area") not in PRODUCT_AREAS or d.get("product_area") == "none":
            d["product_area"] = "other"
    else:
        d["product_area"] = "none"

    # Normalize service_area
    if label == "service":
        if d.get("service_area") not in SERVICE_AREAS or d.get("service_area") == "none":
            d["service_area"] = "other"
    else:
        d["service_area"] = "none"

    # Normalize reason_codes type
    if not isinstance(d.get("reason_codes"), list):
        d["reason_codes"] = []

    return d
