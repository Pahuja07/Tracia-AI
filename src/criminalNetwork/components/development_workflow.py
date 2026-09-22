"""Local, frontend-facing pipeline execution for uploaded development cases."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.criminalNetwork.components.case_upload_handler import CaseUploadHandler
from src.criminalNetwork.components.document_processor import DocumentProcessor
from src.criminalNetwork.components.case_understanding import CaseUnderstanding
from src.criminalNetwork.components.dynamic_schema import DynamicExtractionSchema
from src.criminalNetwork.components.evidence_extraction import EvidenceExtraction
from src.criminalNetwork.components.tabular_model import TabularModelBuilder
from src.criminalNetwork.components.relationship_extraction import RelationshipExtraction
from src.criminalNetwork.components.entity_resolution import EntityResolution
from src.criminalNetwork.components.rag_pipeline import RAGPipeline
from src.criminalNetwork.config.configuration import ConfigurationManager


def _read_rows(path: Path, case_id: str) -> list[dict]:
    if not path.exists() or not path.stat().st_size:
        return []
    try:
        frame = pd.read_csv(path).fillna("")
    except pd.errors.EmptyDataError:
        return []
    if "case_id" in frame:
        frame = frame[frame["case_id"].astype(str) == str(case_id)]
    return json.loads(frame.to_json(orient="records"))


def run_uploaded_case_pipeline(original_filename: str) -> dict:
    """Run auditable local stages and return outputs the frontend can render.

    Neo4j writes and network analytics are intentionally excluded: development
    uploads must not mutate a shared production graph. Those stages remain in
    the production `main.py` pipeline.
    """
    manager = ConfigurationManager()
    upload_config = manager.get_case_upload_config()
    records = CaseUploadHandler(upload_config).upload_cases()
    candidates = [
        record for record in records
        if record["original_filename"] == original_filename
        and record.get("processing_status") == "pending"
    ]
    if not candidates:
        raise ValueError("The upload was already processed or could not be registered.")
    case_id = candidates[-1]["case_id"]

    documents = DocumentProcessor(manager.get_document_processing_config()).run()
    if documents.empty or case_id not in set(documents["case_id"].astype(str)):
        raise ValueError("The uploaded file produced no processable text.")

    stages = ["upload", "document_processing"]
    CaseUnderstanding(manager.get_case_understanding_config()).run(); stages.append("case_understanding")
    DynamicExtractionSchema(manager.get_dynamic_schema_config()).run(); stages.append("dynamic_schema")
    EvidenceExtraction(manager.get_evidence_extraction_config()).run(); stages.append("evidence_extraction")
    TabularModelBuilder(manager.get_tabular_model_config()).run(); stages.append("tabular_model")
    RelationshipExtraction(manager.get_relationship_extraction_config()).build_relationships(); stages.append("relationship_extraction")
    EntityResolution(manager.get_entity_resolution_config()).run(); stages.append("entity_resolution")
    RAGPipeline(manager.get_rag_pipeline_config()).run(); stages.append("chroma_indexing")

    # Mark exactly this development case complete, retaining the same manifest
    # state model as the production upload pipeline.
    manifest = json.loads(upload_config.manifest_path.read_text(encoding="utf-8"))
    for record in manifest:
        if record["case_id"] == case_id:
            record["processing_status"] = "completed"
    upload_config.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "case_id": case_id,
        "original_filename": original_filename,
        "environment": "development",
        "completed_stages": stages,
        "not_run": ["neo4j_graph_build", "graph_analytics", "production_graph_writes"],
        "generated": {
            "documents": _read_rows(manager.get_document_processing_config().documents_path, case_id),
            "case_profile": [profile for profile in json.loads(manager.get_case_understanding_config().profiles_path.read_text(encoding="utf-8")) if profile.get("case_id") == case_id],
            "schema": [schema for schema in json.loads(manager.get_dynamic_schema_config().schemas_path.read_text(encoding="utf-8")) if schema.get("case_id") == case_id],
            "entities": _read_rows(manager.get_tabular_model_config().entities_path, case_id),
            "evidence": _read_rows(manager.get_tabular_model_config().evidence_path, case_id),
            "relationships": _read_rows(manager.get_relationship_extraction_config().output_path, case_id),
        },
    }
