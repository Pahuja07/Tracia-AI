"""Convert registered source files into page-aware, auditable text records."""
import csv
import json
from pathlib import Path
import pandas as pd
from src.criminalNetwork.entity.config_entity import DocumentProcessingConfig


class DocumentProcessor:
    def __init__(self, config: DocumentProcessingConfig): self.config = config

    def _pages(self, path: Path):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader
            return [(i + 1, page.extract_text() or "") for i, page in enumerate(PdfReader(path).pages)]
        if suffix in {".png", ".jpg", ".jpeg"}:
            import pytesseract
            from PIL import Image
            return [(1, pytesseract.image_to_string(Image.open(path)))]
        if suffix == ".csv":
            with path.open(encoding="utf-8", errors="ignore", newline="") as fh:
                return [(1, "\n".join(" | ".join(row) for row in csv.reader(fh)))]
        if suffix == ".json":
            return [(1, json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False))]
        return [(1, path.read_text(encoding="utf-8", errors="ignore"))]

    def run(self):
        manifest = json.loads(self.config.manifest_path.read_text(encoding="utf-8"))
        pending = [item for item in manifest if item.get("processing_status") == "pending"]
        rows = []
        for item in pending:
            for page_number, text in self._pages(Path(item["stored_path"])):
                rows.append({**item, "page_number": page_number, "text": text.strip()})
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        batch = pd.DataFrame(rows, columns=["case_id", "original_filename", "stored_path", "file_type", "sha256", "processing_status", "page_number", "text"])
        batch_path = self.config.output_dir / "new_documents.csv"
        batch.to_csv(batch_path, index=False)
        existing = pd.read_csv(self.config.documents_path) if self.config.documents_path.exists() else pd.DataFrame()
        result = pd.concat([existing, batch], ignore_index=True).drop_duplicates(subset=["case_id", "page_number"])
        result.to_csv(self.config.documents_path, index=False)
        for item in manifest:
            if item.get("processing_status") == "pending": item["processing_status"] = "document_processed"
        self.config.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return batch
