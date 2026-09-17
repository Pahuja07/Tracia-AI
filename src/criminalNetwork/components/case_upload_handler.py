import hashlib
import json
import shutil
from pathlib import Path

from src.criminalNetwork.entity.config_entity import CaseUploadConfig
from src.criminalNetwork.utils.logger import logger


class CaseUploadHandler:
    """Validates and stores case files of supported types for later extraction."""

    SUPPORTED_TYPES = {".txt", ".csv", ".json", ".pdf", ".png", ".jpg", ".jpeg"}

    def __init__(self, config: CaseUploadConfig):
        self.config = config

    def upload_cases(self) -> list[dict]:
        records = []
        if self.config.manifest_path.exists():
            records = json.loads(self.config.manifest_path.read_text(encoding="utf-8"))
            # Older manifests did not track state; their cases already exist in artifacts.
            for record in records:
                record.setdefault("processing_status", "completed")
        known_hashes = {record["sha256"] for record in records}
        for source in sorted(self.config.input_dir.iterdir()):
            if not source.is_file():
                continue
            suffix = source.suffix.lower()
            if suffix not in self.SUPPORTED_TYPES:
                logger.warning("Skipping unsupported case file: %s", source.name)
                continue

            content_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            if content_hash in known_hashes:
                continue
            case_id = f"case_{content_hash[:12]}"
            destination = self.config.upload_dir / f"{case_id}{suffix}"
            shutil.copy2(source, destination)
            records.append({
                "case_id": case_id,
                "original_filename": source.name,
                "stored_path": str(destination),
                "file_type": suffix.lstrip("."),
                "sha256": content_hash,
                "processing_status": "pending",
            })
            known_hashes.add(content_hash)

        self.config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        logger.info("Registered %d new case file(s); manifest: %s", sum(record.get("processing_status") == "pending" for record in records), self.config.manifest_path)
        return records
