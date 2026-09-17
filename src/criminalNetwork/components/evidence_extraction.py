"""Schema-constrained extraction with immutable source references for every fact."""
import json
import re
import pandas as pd
from src.criminalNetwork.entity.config_entity import EvidenceExtractionConfig


class EvidenceExtraction:
    PATTERNS = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "PHONE": r"\b(?!\d{4}-\d{2}-\d{2})(?:\+?\d[\d .()-]{7,}\d)\b",
        "DATE": r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
        "BANK_ACCOUNT": r"\b(?:account|a/c)[ :#-]*([0-9]{6,18})\b",
        "VEHICLE": r"\b[A-Z]{2}[ -]?\d{1,2}[ -]?[A-Z]{1,3}[ -]?\d{4}\b",
        "CASE_REFERENCE": r"\b(?:FIR|CASE|CR|REPORT)[ -]?[A-Z0-9/-]{3,}\b",
        "PERSON": r"(?im)^\s*(?:person|name|accused|victim|suspect)\s*:\s*([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+)",
        "LOCATION": r"(?im)^\s*(?:location|address|place|city)\s*:\s*(.+)$",
    }
    def __init__(self, config: EvidenceExtractionConfig): self.config = config
    def run(self):
        batch_path = self.config.documents_path.parent / "new_documents.csv"
        docs = pd.read_csv(batch_path).fillna("") if batch_path.exists() else pd.DataFrame()
        if docs.empty:
            return pd.DataFrame(), pd.DataFrame()
        schemas = {row["case_id"]: row for row in json.loads(self.config.schemas_path.read_text(encoding="utf-8"))}
        entity_rows, evidence_rows = [], []
        for doc in docs.itertuples(index=False):
            allowed = set(schemas[doc.case_id]["entity_types"])
            text = str(doc.text)
            for entity_type, pattern in self.PATTERNS.items():
                if entity_type not in allowed: continue
                for match in re.finditer(pattern, text):
                    value = match.group(1) if match.lastindex else match.group(0)
                    value = value.strip()
                    if not value: continue
                    evidence_id = f"EV-{doc.case_id}-{doc.page_number}-{match.start()}"
                    common = {"case_id": doc.case_id, "entity_type": entity_type, "entity_value": value, "source_file": doc.original_filename, "page_number": doc.page_number, "text_span": match.group(0), "char_start": match.start(), "char_end": match.end(), "extraction_confidence": 0.85, "evidence_id": evidence_id, "sha256": doc.sha256}
                    entity_rows.append(common)
                    evidence_rows.append({**common, "evidence_type": "entity"})
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.entities_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        entities = pd.DataFrame(entity_rows).drop_duplicates(subset=["case_id", "entity_type", "entity_value", "evidence_id"])
        evidence = pd.DataFrame(evidence_rows)
        previous_entities = pd.read_csv(self.config.entities_path) if self.config.entities_path.exists() else pd.DataFrame()
        previous_evidence = pd.read_csv(self.config.evidence_path) if self.config.evidence_path.exists() else pd.DataFrame()
        entities = pd.concat([previous_entities, entities], ignore_index=True).drop_duplicates(subset=["evidence_id"])
        evidence = pd.concat([previous_evidence, evidence], ignore_index=True).drop_duplicates(subset=["evidence_id"])
        entities.to_csv(self.config.entities_path, index=False)
        evidence.to_csv(self.config.evidence_path, index=False)
        return entities, evidence
