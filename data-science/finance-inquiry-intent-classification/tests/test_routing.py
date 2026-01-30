from src.routing import derive_routing, normalize_areas


def test_routing_product_high_confidence_goes_to_sales():
    classification = {
        "label": "product",
        "confidence": 0.9,
        "product_area": "mortgage",
        "service_area": "documents",  # intentionally wrong -> should be normalized
    }

    normalized = normalize_areas(classification)
    routing = derive_routing(normalized, threshold=0.75)

    assert normalized["label"] == "product"
    assert routing["handoff"] == "sales"
    assert routing["needs_human"] is False
    assert normalized["service_area"] == "none"


def test_routing_service_low_confidence_goes_to_clearing():
    classification = {
        "label": "service",
        "confidence": 0.6,
        "product_area": "mortgage",  # intentionally wrong -> should be normalized
        "service_area": "documents",
    }

    normalized = normalize_areas(classification)
    routing = derive_routing(normalized, threshold=0.75)

    assert normalized["label"] == "service"
    assert routing["handoff"] == "clearing"
    assert routing["needs_human"] is True
    assert normalized["product_area"] == "none"


def test_routing_unknown_label_forced_to_unclear_and_clearing():
    classification = {
        "label": "products",  # invalid label
        "confidence": 0.99,
        "product_area": "mortgage",
        "service_area": "documents",
    }

    normalized = normalize_areas(classification)
    routing = derive_routing(normalized, threshold=0.75)

    # derive_routing does not mutate the input; it treats unknown label as unclear internally
    assert routing["needs_human"] is True
    assert routing["handoff"] == "clearing"

    # normalize_areas enforces: not product -> product_area none; not service -> service_area none
    assert normalized["product_area"] == "none"
    assert normalized["service_area"] == "none"
