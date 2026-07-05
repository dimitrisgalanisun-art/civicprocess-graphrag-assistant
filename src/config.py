from pathlib import Path
import os
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PRIVATE_DIR = PROJECT_ROOT / "data_source" / "private"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"

RAW_MARKDOWN_FILE = DATA_PRIVATE_DIR / "municipality_applications.md"
QA_JSONL_FILE = DATA_PRIVATE_DIR / "municipality_qa.jsonl"

load_dotenv(PROJECT_ROOT / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


def validate_private_paths():
    DATA_PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_MARKDOWN_FILE.exists():
        raise FileNotFoundError(
            f"Private markdown file not found: {RAW_MARKDOWN_FILE}"
        )


def validate_env():
    missing = []

    required = {
        "HF_TOKEN": HF_TOKEN,
        "NEO4J_URI": NEO4J_URI,
        "NEO4J_USERNAME": NEO4J_USERNAME,
        "NEO4J_PASSWORD": NEO4J_PASSWORD,
    }

    for key, value in required.items():
        if not value:
            missing.append(key)

    if missing:
        raise EnvironmentError(
            "Missing required environment variables: "
            + ", ".join(missing)
        )