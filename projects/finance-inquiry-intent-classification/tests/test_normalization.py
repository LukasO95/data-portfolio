from src.normalization import normalize_llm_output


def test_normalize_llm_output_sets_other_for_invalid_product_area():
    raw = {
        "label": "product",
        "confidence": 0.9,
        "product_area": "modernization_financing",  # invalid
        "service_area": "documents",
        "reason_codes": "modernization_financing",
        "schema_version": "v1",
    }

    out = normalize_llm_output(raw)

    assert out["product_area"] == "other"
    assert out["service_area"] == "none"
    assert isinstance(out["reason_codes"], list)
