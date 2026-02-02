# Personally Identifiable Information (PII) masking for text data (non-exhaustive)

import re

def mask_pii(text):
    s = text

    # Email
    s = re.sub(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "<EMAIL>", s, flags=re.IGNORECASE
    )

    # URL
    s = re.sub(
        r"\bhttps?://[^\s]+\b|\bwww\.[^\s]+\b",
        "<URL>", s, flags=re.IGNORECASE
    )

    # IBAN 
    s = re.sub(
        r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b",
        "<IBAN>", s, flags=re.IGNORECASE
    )

    # ID (e.g., Contract Number, Customer Number)
    s = re.sub(
        r"\b(vertrags(?:nummer|nr)?|kunden(?:nummer|nr)?|ticket(?:nummer|nr)?|"
        r"antrags(?:nummer|nr)?|vorgangs(?:nummer|nr)?)\s*[:#]?\s*\w{4,}\b",
        "<ID>", s, flags=re.IGNORECASE
    )

    # Phone
    s = re.sub(
        r"(?:(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{2,5}\)?[\s\-]?)\d{3,}"
        r"(?:[\s\-]?\d{2,})*)",
        "<PHONE>", s
    )

    return s