from src.pii_masking import mask_pii

def test_mask_pii_masks_common_patterns():
    text = (
        "Bitte kontaktieren Sie mich unter max.mustermann@example.com oder +49 171 1234567. "
        "IBAN: DE44500105175407324931. Vertragsnummer: ABCD1234."
    )
    masked = mask_pii(text)

    assert "<EMAIL>" in masked
    assert "<PHONE>" in masked
    assert "<IBAN>" in masked
    assert "<ID>" in masked
