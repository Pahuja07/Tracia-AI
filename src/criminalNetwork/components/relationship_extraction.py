import pandas as pd
import re

from src.criminalNetwork.entity.config_entity import RelationshipExtractionConfig
from src.criminalNetwork.utils.common import read_yaml
from src.criminalNetwork.utils.logger import logger


class RelationshipExtraction:
    """Build graph-ready relationships from extracted case entities."""

    def __init__(self, config: RelationshipExtractionConfig):
        self.config = config

    @staticmethod
    def _structured_relationships(case_id: str, text: str):
        people = {m.group(1): m.group(2) for m in re.finditer(r"(?m)^\s*(P\d{3})\s*-\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)\s*-", text)}
        companies = {m.group(1): m.group(2).strip() for m in re.finditer(r"(?m)^\s*(C\d{3})\s*-\s*([^\n-]+?)\s*-", text)}
        rows = []
        for match in re.finditer(r"(?m)^\s*T\d{3},([^,]+),([^,]+),(\d+),(\d{4}-\d{2}-\d{2}),([^\n]+)$", text):
            source, target = companies.get(match.group(1), match.group(1)), companies.get(match.group(2), match.group(2))
            rows.append({"case_id": case_id, "source_entity": source, "source_type": "ORGANIZATION", "relation": "TRANSFERRED_TO", "target_entity": target, "target_type": "ORGANIZATION", "source_evidence_id": "", "target_evidence_id": "", "extraction_confidence": .98})
        for match in re.finditer(r"(?m)^\s*COM\d+,(P\d+),(P\d+),(\d{4}-\d{2}-\d{2}),([^,]+),", text):
            source, target = people.get(match.group(1), match.group(1)), people.get(match.group(2), match.group(2))
            rows.append({"case_id": case_id, "source_entity": source, "source_type": "PERSON", "relation": "CALLS", "target_entity": target, "target_type": "PERSON", "source_evidence_id": "", "target_evidence_id": "", "extraction_confidence": .98})
        for match in re.finditer(r"(?m)^\s*PR\d+,([^,]+),(\d+),(\d{4}-\d{2}-\d{2}),(P\d+)\s*$", text):
            buyer = companies.get(match.group(1), people.get(match.group(1), match.group(1)))
            buyer_type = "ORGANIZATION" if match.group(1) in companies else "PERSON"
            broker = people.get(match.group(4), match.group(4))
            rows.append({"case_id": case_id, "source_entity": broker, "source_type": "PERSON", "relation": "ASSOCIATED_WITH", "target_entity": buyer, "target_type": buyer_type, "source_evidence_id": "", "target_evidence_id": "", "extraction_confidence": .95})
        return rows

    def build_relationships(self) -> pd.DataFrame:
        if not self.config.common_entities_path.exists():
            raise FileNotFoundError(f"Common entities file not found: {self.config.common_entities_path}")

        entities = pd.read_csv(self.config.common_entities_path)
        batch_path = self.config.common_entities_path.parent.parent / "document_processing" / "new_documents.csv"
        if batch_path.exists():
            new_case_ids = set(pd.read_csv(batch_path)["case_id"])
            entities = entities[entities["case_id"].isin(new_case_ids)]
        if entities.empty:
            return pd.DataFrame()
        rules = read_yaml(self.config.relationship_mapping_file).get("relationships", [])
        rows: list[dict] = []

        document_text = {}
        if batch_path.exists():
            documents = pd.read_csv(batch_path).fillna("")
            document_text = documents.groupby("case_id")["text"].agg("\n".join).to_dict()

        for case_id, case_entities in entities.groupby("case_id"):
            for rule in rules:
                source_type, target_type = rule["source"], rule["target"]
                sources = [{"entity_value": str(case_id), "evidence_id": "", "extraction_confidence": 1.0}] if source_type == "CASE" else case_entities[case_entities["entity_type"] == source_type].to_dict("records")
                targets = case_entities[case_entities["entity_type"] == target_type].to_dict("records")
                for source in sources:
                    for target in targets:
                        rows.append({"case_id": case_id, "source_entity": source["entity_value"], "source_type": source_type, "relation": rule["relation"], "target_entity": target["entity_value"], "target_type": target_type, "source_evidence_id": source.get("evidence_id", ""), "target_evidence_id": target.get("evidence_id", ""), "extraction_confidence": min(float(source.get("extraction_confidence", 1.0)), float(target.get("extraction_confidence", 1.0)))})
            rows.extend(self._structured_relationships(case_id, document_text.get(case_id, "")))

        columns = ["case_id", "source_entity", "source_type", "relation", "target_entity", "target_type", "source_evidence_id", "target_evidence_id", "extraction_confidence"]
        result = pd.DataFrame(rows, columns=columns).drop_duplicates()
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        existing = pd.read_csv(self.config.output_path) if self.config.output_path.exists() else pd.DataFrame(columns=columns)
        result = pd.concat([existing, result], ignore_index=True).drop_duplicates(subset=["case_id", "source_entity", "relation", "target_entity"])
        result.to_csv(self.config.output_path, index=False)
        logger.info("Created %d relationship record(s): %s", len(result), self.config.output_path)
        return result
