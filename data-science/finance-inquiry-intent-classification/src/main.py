import argparse
import json
import os

from .pipeline import classify_and_prepare_crm
from .config import DEFAULT_THRESHOLD, DEFAULT_MODEL

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["mock", "openai"],
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)

    args = parser.parse_args()

    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider

    result = classify_and_prepare_crm(
        raw_text=args.text,
        threshold=args.threshold,
        model=args.model,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
