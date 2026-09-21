"""Evidence-grounded frontend API backed by TRACIA pipeline artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.criminalNetwork.components.spatial_intelligence import SpatialIntelligenceService
from src.criminalNetwork.config.configuration import ConfigurationManager

router = APIRouter(prefix="/api", tags=["TRACIA"])


def _manager(): return ConfigurationManager()
def _spatial(): return SpatialIntelligenceService(_manager().get_spatial_intelligence_config())
def _read(path: Path):
    try: return pd.read_csv(path).fillna("") if path.exists() and path.stat().st_size else pd.DataFrame()
    except pd.errors.EmptyDataError: return pd.DataFrame()
def _records(frame): return json.loads(frame.to_json(orient="records")) if not frame.empty else []
def _one(frame, column, value, label):
    rows = frame[frame[column].astype(str) == str(value)] if not frame.empty and column in frame else pd.DataFrame()
    if rows.empty: raise HTTPException(404, f"{label} '{value}' was not found")
    return _records(rows.head(1))[0]
def _paths():
    manager, spatial, tabular = _manager(), _manager().get_spatial_intelligence_config(), _manager().get_tabular_model_config()
    return {"cases": tabular.cases_path, "evidence": tabular.evidence_path, "relationships": spatial.relationships_file,
            "resolved": spatial.resolved_entities_file, "centrality": spatial.centrality_file, "communities": spatial.community_file,
            "leads": spatial.lead_scores_file, "profiles": manager.get_case_understanding_config().profiles_path}
def _case_relationships(case_id):
    frame = _read(_paths()["relationships"])
    return frame[frame["case_id"].astype(str) == str(case_id)] if not frame.empty else frame
def _case_evidence(case_id):
    frame = _read(_paths()["evidence"])
    return frame[frame["case_id"].astype(str) == str(case_id)] if not frame.empty else frame
def _cyber_unavailable():
    raise HTTPException(501, "Cyberattack APIs are unavailable: no validated cyberattack dataset or similarity output is configured.")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    case_id: str | None = None
    entity_focus: str | None = None


@router.get("/cases", tags=["Cases"])
def list_cases(): return {"cases": _records(_read(_paths()["cases"]))}

@router.get("/cases/{case_id}", tags=["Cases"])
def get_case(case_id: str): return _one(_read(_paths()["cases"]), "case_id", case_id, "Case")

@router.get("/cases/{case_id}/summary", tags=["Cases"])
def case_summary(case_id: str):
    case = get_case(case_id); profiles = json.loads(_paths()["profiles"].read_text(encoding="utf-8")) if _paths()["profiles"].exists() else []
    return {"case": case, "case_understanding": next((p for p in profiles if str(p.get("case_id")) == case_id), {}), "evidence_count": len(_case_evidence(case_id)), "relationship_count": len(_case_relationships(case_id))}

@router.get("/cases/{case_id}/evidence", tags=["Cases"])
def case_evidence(case_id: str):
    get_case(case_id); return {"case_id": case_id, "evidence": _records(_case_evidence(case_id))}


@router.get("/map/case/{case_id}", tags=["Map"])
def case_map(case_id: str, entity_id: str | None = None, entity_type: str | None = None,
             start_date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"), end_date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
    if start_date and end_date and start_date > end_date: raise HTTPException(422, "start_date must not be after end_date")
    geojson = _spatial().map_data(case_id, entity_id, entity_type, start_date, end_date)
    relationships = _case_relationships(case_id); ids = set(relationships.get("source_entity_id", pd.Series(dtype=str)).astype(str)) | set(relationships.get("target_entity_id", pd.Series(dtype=str)).astype(str))
    entities = _read(_paths()["resolved"]); entities = entities[entities["entity_id"].astype(str).isin(ids)] if not entities.empty else entities
    return {"case_id": case_id, "entities": _records(entities), "locations": geojson["unmapped_locations"], "geojson": geojson}

@router.get("/map/entities", tags=["Map"])
def map_entities(case_id: str | None = None, entity_type: str | None = None, start_date: str | None = None, end_date: str | None = None):
    entities = _read(_paths()["resolved"])
    if entity_type and not entities.empty: entities = entities[entities["entity_type"].astype(str).str.lower() == entity_type.lower()]
    if case_id:
        relationships = _case_relationships(case_id); ids = set(relationships.get("source_entity_id", pd.Series(dtype=str)).astype(str)) | set(relationships.get("target_entity_id", pd.Series(dtype=str)).astype(str))
        entities = entities[entities["entity_id"].astype(str).isin(ids)] if not entities.empty else entities
    return {"entities": _records(entities), "filters": {"case_id": case_id, "entity_type": entity_type, "start_date": start_date, "end_date": end_date}}

@router.get("/map/entity/{entity_id}", tags=["Map"])
def map_entity(entity_id: str, case_id: str = Query(...)):
    entity = get_entity(entity_id); geojson = _spatial().map_data(case_id, entity_id=entity_id)
    locations = [feature["properties"] | {"longitude": feature["geometry"]["coordinates"][0], "latitude": feature["geometry"]["coordinates"][1]} for feature in geojson["features"]] + geojson["unmapped_locations"]
    return {"entity_id": entity_id, "name": entity["name"], "type": entity["type"], "locations": locations}

@router.get("/map/location/{location_id}", tags=["Map"])
def map_location(location_id: str): return _one(_read(_paths()["resolved"]), "entity_id", location_id, "Location")

@router.get("/map/cyber-attacks", tags=["Map"])
def cyber_map(start_date: str | None = None, end_date: str | None = None): _cyber_unavailable()


@router.get("/entities/{entity_id}", tags=["Entities"])
def get_entity(entity_id: str):
    entity = _one(_read(_paths()["resolved"]), "entity_id", entity_id, "Entity")
    rels = entity_relationships(entity_id)["relationships"]
    return {"entity_id": entity_id, "name": entity["canonical_name"], "type": entity["entity_type"], "aliases": [], "cases": sorted({r.get("case_id") for r in rels}), "locations": [], "relationships": rels}

@router.get("/entities/{entity_id}/relationships", tags=["Entities"])
def entity_relationships(entity_id: str):
    _one(_read(_paths()["resolved"]), "entity_id", entity_id, "Entity")
    frame = _read(_paths()["relationships"])
    rows = frame[(frame.get("source_entity_id", pd.Series(dtype=str)).astype(str) == entity_id) | (frame.get("target_entity_id", pd.Series(dtype=str)).astype(str) == entity_id)] if not frame.empty else frame
    return {"entity_id": entity_id, "relationships": _records(rows)}

@router.get("/entities/{entity_id}/evidence", tags=["Entities"])
def entity_evidence(entity_id: str):
    entity = _one(_read(_paths()["resolved"]), "entity_id", entity_id, "Entity"); evidence = _read(_paths()["evidence"])
    rows = evidence[evidence.get("entity_value", pd.Series(dtype=str)).astype(str).str.lower() == str(entity["canonical_name"]).lower()] if not evidence.empty else evidence
    return {"entity_id": entity_id, "evidence": _records(rows)}


@router.get("/graph/case/{case_id}", tags=["Graph"])
def case_graph(case_id: str):
    edges = _case_relationships(case_id); ids = set(edges.get("source_entity_id", pd.Series(dtype=str)).astype(str)) | set(edges.get("target_entity_id", pd.Series(dtype=str)).astype(str)); nodes = _read(_paths()["resolved"])
    return {"case_id": case_id, "nodes": _records(nodes[nodes["entity_id"].astype(str).isin(ids)]) if not nodes.empty else [], "edges": _records(edges)}

@router.get("/graph/entity/{entity_id}", tags=["Graph"])
def entity_graph(entity_id: str, depth: int = Query(1, ge=1, le=3)):
    edges = entity_relationships(entity_id)["relationships"]; ids = {entity_id} | {str(e.get("source_entity_id")) for e in edges} | {str(e.get("target_entity_id")) for e in edges}; nodes = _read(_paths()["resolved"])
    return {"entity_id": entity_id, "depth": depth, "nodes": _records(nodes[nodes["entity_id"].astype(str).isin(ids)]), "edges": edges}


@router.get("/analytics/entity/{entity_id}", tags=["Analytics"])
def entity_analytics(entity_id: str):
    centrality = _one(_read(_paths()["centrality"]), "entity_id", entity_id, "Analytics"); communities = _read(_paths()["communities"]); community = _one(communities, "entity_id", entity_id, "Community") if not communities.empty else {}
    return {"entity_id": entity_id, **centrality, "community_id": community.get("community_id", "")}

@router.get("/analytics/case/{case_id}", tags=["Analytics"])
def case_analytics(case_id: str):
    leads = _read(_paths()["leads"]); return {"case_id": case_id, "investigative_leads": _records(leads[leads["case_id"].astype(str) == case_id]) if not leads.empty else []}

@router.get("/analytics/communities", tags=["Analytics"])
def communities(): return {"communities": _records(_read(_paths()["communities"]))}


@router.get("/cyber-attacks", tags=["Cyberattack"])
@router.get("/cyber-attacks/similarity", tags=["Cyberattack"])
@router.get("/cyber-attacks/{attack_id}", tags=["Cyberattack"])
@router.get("/cyber-attacks/{attack_id}/similar", tags=["Cyberattack"])
def cyber_attacks(attack_id: str | None = None, attack_a: str | None = None, attack_b: str | None = None, start_date: str | None = None, end_date: str | None = None, attack_type: str | None = None): _cyber_unavailable()


@router.get("/evidence/{evidence_id}", tags=["Evidence"])
def get_evidence(evidence_id: str): return _one(_read(_paths()["evidence"]), "evidence_id", evidence_id, "Evidence")

@router.post("/ask", tags=["RAG / LLM"])
def ask(request: AskRequest):
    try:
        from src.criminalNetwork.components.agent import CriminalNetworkAgent
        agent = CriminalNetworkAgent(_manager().get_agent_config())
        try:
            answer = agent.answer_query(request.question, request.entity_focus, request.case_id)
            return {"answer": answer, "case_id": request.case_id, "grounded": True, "answer_mode": agent.last_answer_mode}
        finally: agent.close()
    except Exception as error: raise HTTPException(503, f"RAG/LLM explanation is unavailable: {error}") from error

@router.post("/entities/{entity_id}/explain", tags=["RAG / LLM"])
def explain_entity(entity_id: str):
    entity = get_entity(entity_id); return {"entity": entity, "evidence": entity_evidence(entity_id)["evidence"], "notice": "Evidence-grounded dossier only; not a conclusion of wrongdoing."}

@router.get("/leads/case/{case_id}", tags=["Investigative Leads"])
def case_leads(case_id: str): return case_analytics(case_id)

@router.get("/leads/{entity_id}", tags=["Investigative Leads"])
def entity_leads(entity_id: str):
    frame = _read(_paths()["leads"]); rows = frame[(frame.get("source_entity_id", pd.Series(dtype=str)).astype(str) == entity_id) | (frame.get("target_entity_id", pd.Series(dtype=str)).astype(str) == entity_id)] if not frame.empty else frame
    return {"entity_id": entity_id, "investigative_leads": _records(rows), "notice": "Lead scores prioritize evidence-based review and are not guilt scores."}

# Compatibility with the earlier spatial API used by existing frontend work.
@router.get("/v1/map/cases", include_in_schema=False)
def legacy_cases(): return {"cases": _spatial().case_ids()}
@router.get("/v1/map/cases/{case_id}", include_in_schema=False)
def legacy_case_map(case_id: str): return _spatial().map_data(case_id)
