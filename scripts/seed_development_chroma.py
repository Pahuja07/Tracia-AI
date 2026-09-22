"""Seed only the fictional development case into the isolated Chroma collection.

Run: $env:TRACIA_ENV='development'; python scripts/seed_development_chroma.py
"""

import os
import sys
from pathlib import Path

# Allow `python scripts/seed_development_chroma.py` from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.criminalNetwork.components.rag_pipeline import RAGPipeline
from src.criminalNetwork.config.configuration import ConfigurationManager


def main() -> None:
    if os.getenv("TRACIA_ENV", "").lower() != "development":
        raise RuntimeError("Refusing to seed outside development mode. Set TRACIA_ENV=development.")
    config = ConfigurationManager().get_rag_pipeline_config()
    print(f"Loading fictional documents from {config.input_documents_dir}", flush=True)
    if not config.input_documents_dir.exists() or not list(config.input_documents_dir.glob("*.txt")):
        raise RuntimeError("No development .txt documents were found to seed.")
    print(f"Indexing into Chroma collection {config.chroma_collection_name}", flush=True)
    RAGPipeline(config).run()
    print(f"Seeded {config.chroma_collection_name} from {config.input_documents_dir}", flush=True)


if __name__ == "__main__":
    main()
