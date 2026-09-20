import sys
from difflib import SequenceMatcher
import pandas as pd

from src.criminalNetwork.entity.config_entity import EntityResolutionConfig
from src.criminalNetwork.utils.logger import logger
from src.criminalNetwork.utils.exception import CriminalNetworkException


class EntityResolution:
    def __init__(self, config: EntityResolutionConfig):
        self.config = config
        self.resolved_entities = {}   # (canonical_name, entity_type) -> entity data
        self.entity_id_counter = 0
        self.mapping_records = []

    def _normalize(self, name: str) -> str:
        return " ".join(str(name).strip().lower().split())

    def _initial_match(self, name1: str, name2: str) -> bool:
        """'Rahul Sharma' vs 'R. Sharma' jaise cases handle karta hai."""
        n1 = name1.replace(".", "").split()
        n2 = name2.replace(".", "").split()
        if not n1 or not n2:
            return False
        if n1[-1] != n2[-1]:          # last name match
            return False
        return n1[0][0] == n2[0][0]   # first-initial match

    def _find_existing_match(self, raw_name: str, entity_type: str):
        norm_name = self._normalize(raw_name)

        # 1. exact match (case-insensitive)
        for (canonical, canonical_type), data in self.resolved_entities.items():
            if canonical_type == entity_type and self._normalize(canonical) == norm_name:
                return canonical, "exact"

        # Identifier-like values must never be fuzzy-merged: similar dates,
        # account numbers, and phone numbers denote distinct observations.
        if entity_type.upper() in {"DATE", "PHONE", "BANK_ACCOUNT", "CASE_REFERENCE", "TRANSACTION", "COMMUNICATION", "PROPERTY"}:
            return None, None

        # 2. fuzzy match
        best_score, best_canonical = 0, None
        for (canonical, canonical_type), data in self.resolved_entities.items():
            if canonical_type != entity_type:
                continue
            score = int(SequenceMatcher(None, self._normalize(canonical), norm_name).ratio() * 100)
            if score > best_score:
                best_score, best_canonical = score, canonical

        if best_canonical and best_score >= self.config.fuzzy_threshold:
            return best_canonical, f"fuzzy ({best_score}%)"

        # 3. initial/abbreviation match (Person only)
        if entity_type.lower() == "person":
            for (canonical, canonical_type), data in self.resolved_entities.items():
                if canonical_type == entity_type and self._initial_match(
                    norm_name, self._normalize(canonical)
                ):
                    return canonical, "initial-match"

        return None, None

    def resolve_entity(self, raw_name: str, entity_type: str) -> str:
        try:
            match, match_type = self._find_existing_match(raw_name, entity_type)

            if match:
                entity_id = self.resolved_entities[(match, entity_type)]["entity_id"]
                logger.info(f"Resolved '{raw_name}' -> '{match}' ({match_type}), id={entity_id}")
                if match_type != "exact":
                    logger.warning(
                        f"Possible duplicate merged: '{raw_name}' -> '{match}' via {match_type}. "
                        f"Manual review recommended."
                    )
            else:
                self.entity_id_counter += 1
                entity_id = f"E{self.entity_id_counter:05d}"
                self.resolved_entities[(raw_name, entity_type)] = {"entity_id": entity_id, "entity_type": entity_type}
                match_type = "new"
                logger.info(f"New entity created: '{raw_name}' -> id={entity_id}")

            self.mapping_records.append({
                "raw_name": raw_name,
                "entity_type": entity_type,
                "resolved_entity_id": entity_id,
                "match_type": match_type,
            })
            return entity_id
        except Exception as e:
            raise CriminalNetworkException(e, sys) from e

    def resolve_relationships(self, mapping_df: pd.DataFrame) -> pd.DataFrame:
        """relationships.csv me source/target names ko resolved entity_id se replace karta hai."""
        try:
            rel_df = pd.read_csv(self.config.input_relationships_file)
            name_type_to_id = {
                (row.raw_name, row.entity_type): row.resolved_entity_id
                for row in mapping_df.itertuples(index=False)
            }

            rel_df["source_entity_id"] = rel_df.apply(
                lambda row: name_type_to_id.get((row["source_entity"], row["source_type"])), axis=1
            )
            rel_df["target_entity_id"] = rel_df.apply(
                lambda row: name_type_to_id.get((row["target_entity"], row["target_type"])), axis=1
            )

            unmatched = rel_df[rel_df["source_entity_id"].isna() | rel_df["target_entity_id"].isna()]
            if not unmatched.empty:
                logger.warning(f"{len(unmatched)} relationship rows could not be fully resolved.")

            rel_df.to_csv(self.config.resolved_relationships_file, index=False)
            logger.info(f"Resolved relationships saved to {self.config.resolved_relationships_file}")
            return rel_df
        except Exception as e:
            raise CriminalNetworkException(e, sys) from e

    def run(self):
        try:
            logger.info("Starting entity resolution stage")

            entities_df = pd.read_csv(self.config.input_entities_file)
            entity_name_column = "entity_name" if "entity_name" in entities_df.columns else "entity_value"
            for _, row in entities_df.iterrows():
                self.resolve_entity(row[entity_name_column], row["entity_type"])

            # Relationship rules can introduce endpoints such as CASE that do
            # not appear in the extracted-entity table. Resolve them too so a
            # valid graph edge is never dropped for a missing ID.
            relationships_df = pd.read_csv(self.config.input_relationships_file)
            for _, row in relationships_df.iterrows():
                for name_column, type_column in (("source_entity", "source_type"), ("target_entity", "target_type")):
                    if pd.notna(row[name_column]) and pd.notna(row[type_column]):
                        self.resolve_entity(str(row[name_column]), str(row[type_column]))

            mapping_df = pd.DataFrame(
                self.mapping_records,
                columns=["raw_name", "entity_type", "resolved_entity_id", "match_type"],
            )
            mapping_df.to_csv(self.config.entity_mapping_file, index=False)
            logger.info(f"Entity mapping saved to {self.config.entity_mapping_file}")

            resolved_df = pd.DataFrame(
                [
                    {"entity_id": v["entity_id"], "canonical_name": canonical_name, "entity_type": entity_type}
                    for (canonical_name, entity_type), v in self.resolved_entities.items()
                ],
                columns=["entity_id", "canonical_name", "entity_type"],
            )
            resolved_df.to_csv(self.config.resolved_entities_file, index=False)
            logger.info(f"Resolved entities saved to {self.config.resolved_entities_file}")

            self.resolve_relationships(mapping_df)

            logger.info("Entity resolution stage completed")
            return resolved_df, mapping_df
        except Exception as e:
            raise CriminalNetworkException(e, sys) from e
