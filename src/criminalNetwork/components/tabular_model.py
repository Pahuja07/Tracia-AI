"""Build the audited relational/tabular layer that is the sole input to graph construction."""
import json
import re
import pandas as pd
from src.criminalNetwork.entity.config_entity import TabularModelConfig


class TabularModelBuilder:
    def __init__(self, config: TabularModelConfig): self.config = config

    @staticmethod
    def _normalise(value: str) -> str:
        return re.sub(r"\s+", " ", str(value).strip().lower())

    def run(self):
        batch_path = self.config.documents_path.parent / "new_documents.csv"
        documents = pd.read_csv(batch_path).fillna("") if batch_path.exists() else pd.DataFrame()
        if documents.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        profiles = pd.DataFrame(json.loads(self.config.profiles_path.read_text(encoding="utf-8")))
        case_ids = set(documents["case_id"])
        extracted = pd.read_csv(self.config.extracted_entities_path).fillna("")
        extracted = extracted[extracted["case_id"].isin(case_ids)]
        evidence = pd.read_csv(self.config.extracted_evidence_path).fillna("")
        evidence = evidence[evidence["case_id"].isin(case_ids)]

        cases = documents.groupby("case_id", as_index=False).agg(
            original_filename=("original_filename", "first"), file_type=("file_type", "first"),
            sha256=("sha256", "first"), page_count=("page_number", "max"), document_text_length=("text", lambda values: sum(len(str(value)) for value in values)),
        ).merge(profiles[["case_id", "case_type", "classification_confidence"]], on="case_id", how="left")
        existing_cases = pd.read_csv(self.config.cases_path) if self.config.cases_path.exists() else pd.DataFrame()
        existing_entities = pd.read_csv(self.config.entities_path) if self.config.entities_path.exists() else pd.DataFrame()
        existing_evidence = pd.read_csv(self.config.evidence_path) if self.config.evidence_path.exists() else pd.DataFrame()
        entities = extracted.copy()
        entities["entity_record_id"] = [f"ENT-{index:08d}" for index in range(len(existing_entities) + 1, len(existing_entities) + len(entities) + 1)]
        entities["normalized_value"] = entities["entity_value"].map(self._normalise)
        entities = entities[["entity_record_id", "case_id", "entity_type", "entity_value", "normalized_value", "evidence_id", "source_file", "page_number", "text_span", "char_start", "char_end", "extraction_confidence", "sha256"]]
        evidence = evidence[["evidence_id", "case_id", "evidence_type", "source_file", "page_number", "text_span", "char_start", "char_end", "extraction_confidence", "sha256", "entity_type", "entity_value"]]
        cases = pd.concat([existing_cases, cases], ignore_index=True).drop_duplicates(subset=["case_id"], keep="last")
        entities = pd.concat([existing_entities, entities], ignore_index=True).drop_duplicates(subset=["evidence_id"], keep="last")
        evidence = pd.concat([existing_evidence, evidence], ignore_index=True).drop_duplicates(subset=["evidence_id"], keep="last")
        for path, frame in ((self.config.cases_path, cases), (self.config.entities_path, entities), (self.config.evidence_path, evidence)):
            path.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(path, index=False)
        return cases, entities, evidence
