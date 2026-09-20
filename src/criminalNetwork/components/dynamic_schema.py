import json
from src.criminalNetwork.entity.config_entity import DynamicSchemaConfig


class DynamicExtractionSchema:
    BASE = ["PERSON", "PHONE", "EMAIL", "DATE", "LOCATION", "CASE_REFERENCE"]
    EXTRA = {
        "financial_fraud": ["BANK_ACCOUNT", "TRANSACTION", "ORGANIZATION", "PROPERTY", "CRIME", "COMMUNICATION"],
        "vehicle_crime": ["VEHICLE", "LOCATION", "ORGANIZATION"],
        "violent_crime": ["WEAPON", "VICTIM", "LOCATION", "ORGANIZATION"],
        "communication_case": ["PHONE", "EMAIL", "COMMUNICATION"],
        "general": [],
    }
    def __init__(self, config: DynamicSchemaConfig): self.config = config
    def run(self):
        profiles = json.loads(self.config.profiles_path.read_text(encoding="utf-8"))
        schemas = [{"case_id": p["case_id"], "case_type": p["case_type"], "entity_types": self.BASE + self.EXTRA[p["case_type"]], "relationship_types": ["ASSOCIATED_WITH", "LOCATED_AT", "CALLS", "OWNS", "MEMBER_OF", "USES_ACCOUNT", "TRANSACTED_WITH"]} for p in profiles]
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.schemas_path.write_text(json.dumps(schemas, indent=2), encoding="utf-8")
        return schemas
