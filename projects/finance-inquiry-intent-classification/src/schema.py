# Schema defintion to prevent label drift and ensure consistent LLM output structure

REASON_CODE_ENUM = [
    # product
    "new_mortgage_request",
    "modernization_financing",
    "additional_financing_request",
    "loan_consolidation",
    "subsidy_interest",
    "rate_or_offer_request",
    "savings_inquiry",
    "fixed_deposit_inquiry",
    "home_savings_inquiry",
    # service
    "online_access_issue",
    "document_request",
    "account_statement_request",
    "tax_certificate_request",
    "release_of_charge_request",
    "contract_information_request",
    "payment_issue",
    "complaint",
    # unclear
    "callback_request_only",
    "appointment_without_topic",
    "insufficient_information",
]

CLASSIFICATION_SCHEMA = {
    "name": "FinanceInquiryClassification_v1",
    "strict": True,  # enforce schema during validation
    "schema": {  # actual JSON Schema
        "type": "object",
        "additionalProperties": False,  # no extra fields allowed
        "properties": {
            "label": {"type": "string", "enum": ["product", "service", "unclear"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "product_area": {
                "type": "string",
                "enum": ["mortgage", "savings", "home_savings", "other", "none"],
            },
            "service_area": {
                "type": "string",
                "enum": [
                    "existing_contract",
                    "online_banking",
                    "documents",
                    "payments",
                    "complaints",
                    "other",
                    "none",
                ],
            },
            "reason_codes": {
                "type": "array",
                "items": {"type": "string", "enum": REASON_CODE_ENUM},
                "minItems": 0,
                "maxItems": 3,
            },
            "schema_version": {"type": "string"},
        },
        # LLM contract
        "required": [
            "label",
            "confidence",
            "product_area",
            "service_area",
            "schema_version",
        ],
    },
}
