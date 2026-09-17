import sys
import pandas as pd
from neo4j import GraphDatabase

from src.criminalNetwork.entity.config_entity import GraphBuilderConfig
from src.criminalNetwork.utils.logger import logger
from src.criminalNetwork.utils.exception import CriminalNetworkException
from src.criminalNetwork.utils.common import load_graph_schema, normalize_neo4j_uri


class GraphBuilder:
    def __init__(self, config: GraphBuilderConfig):
        self.config = config
        uri = normalize_neo4j_uri(self.config.neo4j_uri, self.config.trust_self_signed_certificate)
        if self.config.trust_self_signed_certificate and ".neo4j.io" not in uri.lower():
            logger.warning("Neo4j self-signed certificate trust is enabled")
        self.driver = GraphDatabase.driver(
            uri,
            auth=(self.config.neo4j_username, self.config.neo4j_password),
        )
        self.schema = load_graph_schema(self.config.graph_schema_file)
        self.entity_types = self.schema.get("entity_types", {})
        self.relationship_types = self.schema.get("relationship_types", {})
        self.build_log = []
        # entity_id -> entity_type lookup, relationship validation ke liye
        self.entity_type_lookup = {}

    def close(self):
        self.driver.close()

    def _create_constraints(self):
        try:
            with self.driver.session(database=self.config.neo4j_database) as session:
                session.run(
                    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
                    "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE"
                )
            logger.info("Constraint on Entity.entity_id ensured")
        except Exception as e:
            raise CriminalNetworkException(e, sys) from e

    @staticmethod
    def _sanitize_label(entity_type: str) -> str:
        return "".join(ch for ch in str(entity_type).title() if ch.isalnum()) or "Entity"

    def _validate_entity_type(self, entity_type: str) -> str:
        """Schema ke against entity_type check karta hai — case-insensitive match, warna 'Unknown' fallback."""
        normalised_input = "".join(char for char in str(entity_type).lower() if char.isalnum())
        for valid_type in self.entity_types:
            normalised_valid = "".join(char for char in valid_type.lower() if char.isalnum())
            if normalised_valid == normalised_input:
                return valid_type
        logger.warning(f"Entity type '{entity_type}' not in schema — treating as 'Unknown'")
        return "Unknown"

    def _validate_relationship(self, rel_type: str, source_type: str, target_type: str):
        """Schema ke against relationship_type + pair check karta hai. Returns (valid_rel_type, is_valid)."""
        matched_type = None
        for valid_type in self.relationship_types:
            if valid_type.lower() == str(rel_type).strip().lower():
                matched_type = valid_type
                break

        if matched_type is None:
            logger.warning(f"Relationship type '{rel_type}' not in schema — falling back to RELATED_TO")
            return "RELATED_TO", False

        valid_pairs = self.relationship_types[matched_type].get("valid_pairs", [])
        if not valid_pairs:  # empty list = koi bhi pair allowed (jaise RELATED_TO)
            return matched_type, True

        pair_ok = [source_type, target_type] in valid_pairs or [target_type, source_type] in valid_pairs
        if not pair_ok:
            logger.warning(
                f"Invalid pair for '{matched_type}': {source_type} -> {target_type}. "
                f"Allowed pairs: {valid_pairs}. Marking for manual review."
            )
            return matched_type, False

        return matched_type, True

    def _merge_node_tx(self, tx, entity_id, canonical_name, entity_type):
        label = self._sanitize_label(entity_type)
        query = (
            f"MERGE (e:Entity {{entity_id: $entity_id}}) "
            f"SET e.name = $name, e.entity_type = $entity_type "
            f"SET e:{label} "
            f"RETURN e.entity_id AS id"
        )
        return tx.run(query, entity_id=entity_id, name=canonical_name, entity_type=entity_type).single()

    def build_nodes(self):
        try:
            entities_df = pd.read_csv(self.config.resolved_entities_file)
            with self.driver.session(database=self.config.neo4j_database) as session:
                for _, row in entities_df.iterrows():
                    validated_type = self._validate_entity_type(row["entity_type"])
                    self.entity_type_lookup[row["entity_id"]] = validated_type

                    session.execute_write(
                        self._merge_node_tx,
                        row["entity_id"],
                        row["canonical_name"],
                        validated_type,
                    )
                    self.build_log.append({
                        "type": "node",
                        "entity_id": row["entity_id"],
                        "name": row["canonical_name"],
                        "entity_type": validated_type,
                        "status": "merged",
                    })
            logger.info(f"{len(entities_df)} nodes merged into Neo4j")
        except Exception as e:
            raise CriminalNetworkException(e, sys) from e

    def _merge_relationship_tx(self, tx, source_id, target_id, rel_type, properties):
        rel_label = "".join(ch for ch in str(rel_type).upper().replace(" ", "_") if ch.isalnum() or ch == "_") or "RELATED_TO"
        query = (
            "MATCH (a:Entity {entity_id: $source_id}) "
            "MATCH (b:Entity {entity_id: $target_id}) "
            f"MERGE (a)-[r:{rel_label}]->(b) "
            "SET r += $properties "
            "RETURN type(r) AS rel_type"
        )
        return tx.run(
            query, source_id=source_id, target_id=target_id, properties=properties
        ).single()

    def build_relationships(self):
        try:
            rel_df = pd.read_csv(self.config.resolved_relationships_file)

            required_cols = {"source_entity_id", "target_entity_id"}
            if not required_cols.issubset(rel_df.columns):
                raise ValueError(f"resolved_relationships.csv missing columns: {required_cols}")

            rel_type_col = next(
                (column for column in ("relationship_type", "relation") if column in rel_df.columns),
                None,
            )
            skipped, flagged = 0, 0

            with self.driver.session(database=self.config.neo4j_database) as session:
                for _, row in rel_df.iterrows():
                    source_id, target_id = row["source_entity_id"], row["target_entity_id"]

                    if pd.isna(source_id) or pd.isna(target_id):
                        skipped += 1
                        continue

                    source_type = self.entity_type_lookup.get(source_id, "Unknown")
                    target_type = self.entity_type_lookup.get(target_id, "Unknown")
                    raw_rel_type = row[rel_type_col] if rel_type_col and pd.notna(row.get(rel_type_col)) else "RELATED_TO"

                    validated_rel_type, is_valid = self._validate_relationship(raw_rel_type, source_type, target_type)
                    if not is_valid:
                        flagged += 1

                    exclude = {"source_entity", "target_entity", "source_entity_id", "target_entity_id", rel_type_col}
                    properties = {
                        col: row[col] for col in rel_df.columns
                        if col not in exclude and pd.notna(row[col])
                    }
                    properties["schema_valid"] = is_valid

                    session.execute_write(
                        self._merge_relationship_tx,
                        source_id,
                        target_id,
                        validated_rel_type,
                        properties,
                    )
                    self.build_log.append({
                        "type": "relationship",
                        "source": source_id,
                        "target": target_id,
                        "rel_type": validated_rel_type,
                        "schema_valid": is_valid,
                        "status": "merged",
                    })

            if skipped:
                logger.warning(f"{skipped} relationship rows skipped due to unresolved entity_id")
            if flagged:
                logger.warning(f"{flagged} relationships flagged as schema-invalid (schema_valid=False) — manual review needed")
            logger.info(f"{len(rel_df) - skipped} relationships merged into Neo4j")
        except Exception as e:
            raise CriminalNetworkException(e, sys) from e

    def run(self):
        try:
            logger.info("Starting graph builder stage")
            self._create_constraints()
            self.build_nodes()
            self.build_relationships()

            log_df = pd.DataFrame(self.build_log)
            log_df.to_csv(self.config.graph_build_log_file, index=False)
            logger.info(f"Graph build log saved to {self.config.graph_build_log_file}")

            logger.info("Graph builder stage completed")
        except Exception as e:
            raise CriminalNetworkException(e, sys) from e
        finally:
            self.close()
