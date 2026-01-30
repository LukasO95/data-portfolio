from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "classifier_v1.txt"

def load_classifier_prompt():
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8")


if __name__ == "__main__":
    print(load_classifier_prompt())
