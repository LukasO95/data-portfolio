from jsonschema import validate

from .schema import CLASSIFICATION_SCHEMA


def validate_output(data):
    """
    Validates the LLM output against the expected JSON schema.
    Raises ValidationError if invalid.
    """
    validate(instance=data, schema=CLASSIFICATION_SCHEMA["schema"])
