"""Deterministic first-pass case classification; may be replaced by a reviewed LLM classifier."""
import json
import pandas as pd
from src.criminalNetwork.entity.config_entity import CaseUnderstandingConfig


class CaseUnderstanding:
    KEYWORDS = {
        "financial_fraud": ("fraud", "bank", "account", "transaction", "transfer", "money"),
        "vehicle_crime": ("vehicle", "car", "motorcycle", "registration", "theft"),
        "violent_crime": ("murder", "weapon", "assault", "victim", "rape"),
        "communication_case": ("call", "phone", "message", "whatsapp", "email"),
    }
    def __init__(self, config: CaseUnderstandingConfig): self.config = config
    def run(self):
        batch_path = self.config.documents_path.parent / "new_documents.csv"
        docs = pd.read_csv(batch_path).fillna("") if batch_path.exists() else pd.DataFrame()
        if docs.empty:
            return []
        profiles = []
        for case_id, group in docs.groupby("case_id"):
            text = " ".join(group.text.astype(str)).lower()
            scores = {name: sum(word in text for word in words) for name, words in self.KEYWORDS.items()}
            case_type, score = max(scores.items(), key=lambda item: item[1])
            profiles.append({"case_id": case_id, "case_type": case_type if score else "general", "classification_confidence": round(score / max(len(self.KEYWORDS[case_type]), 1), 2), "requirements": [key for key, value in scores.items() if value]})
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        existing = json.loads(self.config.profiles_path.read_text(encoding="utf-8")) if self.config.profiles_path.exists() else []
        all_profiles = {profile["case_id"]: profile for profile in existing}
        all_profiles.update({profile["case_id"]: profile for profile in profiles})
        self.config.profiles_path.write_text(json.dumps(list(all_profiles.values()), indent=2), encoding="utf-8")
        return profiles
