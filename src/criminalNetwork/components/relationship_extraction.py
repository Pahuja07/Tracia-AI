import pandas as pd

from src.criminalNetwork.entity.config_entity import RelationshipExtractionConfig
from src.criminalNetwork.utils.common import read_yaml
from src.criminalNetwork.utils.logger import logger


class RelationshipExtraction:
    """Build graph-ready relationships from extracted case entities."""

    def __init__(self, config: RelationshipExtractionConfig):
        self.config = config

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

        for case_id, case_entities in entities.groupby("case_id"):
            for rule in rules:
                source_type, target_type = rule["source"], rule["target"]
                sources = [{"entity_value": str(case_id), "evidence_id": "", "extraction_confidence": 1.0}] if source_type == "CASE" else case_entities[case_entities["entity_type"] == source_type].to_dict("records")
                targets = case_entities[case_entities["entity_type"] == target_type].to_dict("records")
                for source in sources:
                    for target in targets:
                        rows.append({"case_id": case_id, "source_entity": source["entity_value"], "source_type": source_type, "relation": rule["relation"], "target_entity": target["entity_value"], "target_type": target_type, "source_evidence_id": source.get("evidence_id", ""), "target_evidence_id": target.get("evidence_id", ""), "extraction_confidence": min(float(source.get("extraction_confidence", 1.0)), float(target.get("extraction_confidence", 1.0)))})

        columns = ["case_id", "source_entity", "source_type", "relation", "target_entity", "target_type", "source_evidence_id", "target_evidence_id", "extraction_confidence"]
        result = pd.DataFrame(rows, columns=columns).drop_duplicates()
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        existing = pd.read_csv(self.config.output_path) if self.config.output_path.exists() else pd.DataFrame(columns=columns)
        result = pd.concat([existing, result], ignore_index=True).drop_duplicates(subset=["case_id", "source_entity", "relation", "target_entity"])
        result.to_csv(self.config.output_path, index=False)
        logger.info("Created %d relationship record(s): %s", len(result), self.config.output_path)
        return result
