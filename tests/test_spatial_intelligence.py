import pandas as pd

from src.criminalNetwork.components.spatial_intelligence import SpatialIntelligenceService
from src.criminalNetwork.entity.config_entity import SpatialIntelligenceConfig


def test_spatial_data_keeps_unmapped_documented_location(tmp_path):
    relationships = tmp_path / "relationships.csv"
    pd.DataFrame([{
        "case_id": "case-1", "source_entity": "case-1", "source_type": "CASE",
        "relation": "OCCURRED_IN", "target_entity": "Jaipur", "target_type": "LOCATION",
        "source_entity_id": "E1", "target_entity_id": "E2", "target_evidence_id": "EV1",
        "extraction_confidence": 0.9,
    }]).to_csv(relationships, index=False)
    evidence = tmp_path / "evidence.csv"
    pd.DataFrame([{"evidence_id": "EV1", "text_span": "Location: Jaipur", "source_file": "fir.txt", "page_number": 1, "extraction_confidence": 0.9}]).to_csv(evidence, index=False)
    resolved = tmp_path / "resolved.csv"
    pd.DataFrame([{"entity_id": "E1", "canonical_name": "case-1", "entity_type": "CASE"}]).to_csv(resolved, index=False)
    empty = tmp_path / "empty.csv"; pd.DataFrame().to_csv(empty, index=False)
    service = SpatialIntelligenceService(SpatialIntelligenceConfig(empty, evidence, relationships, resolved, empty, empty, empty))

    result = service.map_data("case-1")

    assert result["features"] == []
    assert result["unmapped_locations"][0]["location"] == "Jaipur"
    assert result["unmapped_locations"][0]["evidence_text"] == "Location: Jaipur"


def test_spatial_data_emits_geojson_only_for_recorded_coordinates(tmp_path):
    relationships = tmp_path / "relationships.csv"
    pd.DataFrame([{
        "case_id": "case-1", "source_entity": "case-1", "source_type": "CASE",
        "relation": "OCCURRED_IN", "target_entity": "Recorded site", "target_type": "LOCATION",
        "source_entity_id": "E1", "target_entity_id": "E2", "target_evidence_id": "EV1",
        "latitude": 26.9124, "longitude": 75.7873,
    }]).to_csv(relationships, index=False)
    evidence = tmp_path / "evidence.csv"; pd.DataFrame([{"evidence_id": "EV1"}]).to_csv(evidence, index=False)
    resolved = tmp_path / "resolved.csv"; pd.DataFrame([{"entity_id": "E1", "canonical_name": "case-1", "entity_type": "CASE"}]).to_csv(resolved, index=False)
    empty = tmp_path / "empty.csv"; pd.DataFrame().to_csv(empty, index=False)
    service = SpatialIntelligenceService(SpatialIntelligenceConfig(empty, evidence, relationships, resolved, empty, empty, empty))

    result = service.map_data("case-1")

    assert result["features"][0]["geometry"]["coordinates"] == [75.7873, 26.9124]
