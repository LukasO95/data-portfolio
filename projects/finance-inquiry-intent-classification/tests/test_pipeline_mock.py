import os

from src.pipeline import classify_and_prepare_crm


def test_pipeline_mock_product_goes_to_sales():
    os.environ["LLM_PROVIDER"] = "mock"

    # Choose a text that the minimal mock reliably treats as product
    res = classify_and_prepare_crm("Ich brauche einen Kredit", threshold=0.75)

    assert res["label"] == "product"
    assert res["handoff"] == "sales"
    assert res["needs_human"] is False


def test_pipeline_mock_service_access_goes_to_support():
    os.environ["LLM_PROVIDER"] = "mock"

    # Choose a text that the minimal mock reliably treats as service
    res = classify_and_prepare_crm(
        "Mein Login ist gesperrt, kein Zugang zum Portal", threshold=0.75
    )

    assert res["label"] == "service"
    assert res["handoff"] == "support"
    assert res["needs_human"] is False


def test_empty_input_returns_unclear():
    os.environ["LLM_PROVIDER"] = "mock"

    res = classify_and_prepare_crm("")
    assert res["label"] == "unclear"
    assert res["needs_human"] is True
    assert res["reason_codes"] == ["insufficient_information"]
