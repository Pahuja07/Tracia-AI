"""Chroma client construction shared by indexing and investigator retrieval."""

from pathlib import Path

import chromadb


def create_chroma_client(mode: str, persist_directory: Path, api_key=None, tenant=None, database=None):
    """Return a durable Chroma client; cloud credentials are never hard-coded."""
    mode = mode.lower()
    if mode == "cloud":
        missing = [name for name, value in {
            "CHROMA_API_KEY": api_key,
            "CHROMA_TENANT": tenant,
            "CHROMA_DATABASE": database,
        }.items() if not value]
        if missing:
            raise ValueError("Chroma Cloud is enabled but missing: " + ", ".join(missing))
        return chromadb.CloudClient(api_key=api_key, tenant=tenant, database=database)
    if mode != "local":
        raise ValueError("CHROMA_MODE must be either 'local' or 'cloud'.")
    persist_directory.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_directory))


def get_evidence_collection(client, name: str):
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine", "purpose": "tracia_case_evidence"},
    )


def get_interactions_collection(client, name: str):
    """Collection that retains API/agent output separately from case evidence."""
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine", "purpose": "tracia_agent_interactions"},
    )
