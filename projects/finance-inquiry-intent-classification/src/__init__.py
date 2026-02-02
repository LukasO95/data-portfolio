from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")