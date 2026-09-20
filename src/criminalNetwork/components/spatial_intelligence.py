"""Evidence-grounded, read-only spatial data for the investigator map.

This component does not geocode, infer travel, or create relationships.  It
adapts the audited tabular and graph-analysis outputs into GeoJSON only where
the underlying record contains coordinates.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from src.criminalNetwork.entity.config_entity import SpatialIntelligenceConfig


class SpatialIntelligenceService:
    def __init__(self, config: SpatialIntelligenceConfig):
        self.config = config

    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        if not path.exists() or not path.stat().st_size:
            return pd.DataFrame()
        try:
            return pd.read_csv(path).fillna("")
        except EmptyDataError:
            return pd.DataFrame()

    @staticmethod
    def _date_in_range(value: str, date_from: str | None, date_to: str | None) -> bool:
        if not value or (not date_from and not date_to):
            return True
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return True  # no date is not grounds to discard a documented fact
        return (not date_from or parsed >= pd.Timestamp(date_from)) and (not date_to or parsed <= pd.Timestamp(date_to))

    @staticmethod
    def _coordinates(*values: Any) -> tuple[float, float] | None:
        """Accept coordinates embedded in an existing record, never a place-name lookup."""
        for latitude_value, longitude_value in zip(values[::2], values[1::2]):
            try:
                latitude, longitude = float(latitude_value), float(longitude_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(latitude) and math.isfinite(longitude) and -90 <= latitude <= 90 and -180 <= longitude <= 180:
                return latitude, longitude
        for value in values:
            if value is None:
                continue
            match = re.search(r"(?<!\d)(-?\d{1,2}(?:\.\d+)?)\s*[,;]\s*(-?\d{1,3}(?:\.\d+)?)(?!\d)", str(value))
            if not match:
                continue
            latitude, longitude = map(float, match.groups())
            if math.isfinite(latitude) and math.isfinite(longitude) and -90 <= latitude <= 90 and -180 <= longitude <= 180:
                return latitude, longitude
        return None

    def case_ids(self) -> list[dict[str, str]]:
        relationships = self._read(self.config.relationships_file)
        if relationships.empty or "case_id" not in relationships:
            return []
        return [{"case_id": str(case_id)} for case_id in sorted(relationships["case_id"].astype(str).unique())]

    def map_data(
        self, case_id: str, entity_id: str | None = None, entity_type: str | None = None,
        date_from: str | None = None, date_to: str | None = None,
    ) -> dict[str, Any]:
        relationships = self._read(self.config.relationships_file)
        evidence = self._read(self.config.evidence_file)
        resolved = self._read(self.config.resolved_entities_file)
        centrality = self._read(self.config.centrality_file)
        communities = self._read(self.config.community_file)
        lead_scores = self._read(self.config.lead_scores_file)
        if relationships.empty:
            return self._empty(case_id, "No relationship data has been produced yet.")

        rows = relationships[relationships["case_id"].astype(str) == str(case_id)].copy()
        if rows.empty:
            return self._empty(case_id, "No documented relationships were found for this case.")
        relation_column = "relation" if "relation" in rows else "relationship_type"
        rows = rows[rows[relation_column].astype(str).str.upper().isin({"LOCATED_AT", "OCCURRED_IN", "SEEN_AT"})]
        if rows.empty:
            return self._empty(case_id, "This case has no documented location relationships.")

        features: list[dict[str, Any]] = []
        unmapped: list[dict[str, Any]] = []
        for row in rows.to_dict("records"):
            subject_is_location = str(row.get("source_type", "")).upper() == "LOCATION"
            location = row.get("source_entity") if subject_is_location else row.get("target_entity")
            subject = row.get("target_entity") if subject_is_location else row.get("source_entity")
            subject_type = row.get("target_type") if subject_is_location else row.get("source_type")
            subject_id = row.get("target_entity_id") if subject_is_location else row.get("source_entity_id")
            if entity_id and str(subject_id) != str(entity_id):
                continue
            if entity_type and str(subject_type).lower() != entity_type.lower():
                continue
            evidence_id = row.get("target_evidence_id") or row.get("source_evidence_id") or ""
            evidence_row = evidence[evidence.get("evidence_id", pd.Series(dtype=str)).astype(str) == str(evidence_id)]
            evidence_data = evidence_row.iloc[0].to_dict() if not evidence_row.empty else {}
            event_date = self._event_date(evidence_data, row)
            if not self._date_in_range(event_date, date_from, date_to):
                continue
            properties = self._properties(row, evidence_data, resolved, centrality, communities, lead_scores, subject_id, case_id)
            coordinates = self._coordinates(row.get("latitude"), row.get("longitude"), evidence_data.get("latitude"), evidence_data.get("longitude"), location)
            record = {"location": str(location), "coordinates_available": coordinates is not None, **properties}
            if coordinates:
                latitude, longitude = coordinates
                features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [longitude, latitude]}, "properties": record})
            else:
                unmapped.append(record)
        return {
            "type": "FeatureCollection", "features": features,
            "case_id": case_id, "unmapped_locations": unmapped,
            "notice": "Markers use only coordinates documented in project records. Textual locations without coordinates are returned separately.",
        }

    @staticmethod
    def _event_date(evidence: dict[str, Any], relationship: dict[str, Any]) -> str:
        for value in (relationship.get("event_date"), evidence.get("event_date"), evidence.get("date")):
            if value:
                return str(value)
        return ""

    @staticmethod
    def _lookup(frame: pd.DataFrame, entity_id: Any) -> dict[str, Any]:
        if frame.empty or "entity_id" not in frame.columns:
            return {}
        rows = frame[frame["entity_id"].astype(str) == str(entity_id)]
        return rows.iloc[0].to_dict() if not rows.empty else {}

    def _properties(self, relationship, evidence, resolved, centrality, communities, lead_scores, entity_id, case_id):
        entity = self._lookup(resolved, entity_id)
        lead = lead_scores[(lead_scores.get("case_id", pd.Series(dtype=str)).astype(str) == str(case_id)) & ((lead_scores.get("source_entity_id", pd.Series(dtype=str)).astype(str) == str(entity_id)) | (lead_scores.get("target_entity_id", pd.Series(dtype=str)).astype(str) == str(entity_id)))] if not lead_scores.empty else pd.DataFrame()
        return {
            "entity_id": str(entity_id or ""),
            "entity_name": str(entity.get("canonical_name") or relationship.get("source_entity") or ""),
            "entity_type": str(entity.get("entity_type") or relationship.get("source_type") or ""),
            "case_id": str(case_id),
            "relationship_context": str(relationship.get("relation") or relationship.get("relationship_type") or ""),
            "date_time": self._event_date(evidence, relationship),
            "evidence_id": str(evidence.get("evidence_id") or relationship.get("target_evidence_id") or relationship.get("source_evidence_id") or ""),
            "evidence_text": str(evidence.get("text_span") or ""),
            "source_document": str(evidence.get("source_file") or ""),
            "page_number": str(evidence.get("page_number") or ""),
            "extraction_confidence": evidence.get("extraction_confidence") or relationship.get("extraction_confidence") or "",
            "analytics": {"centrality": self._lookup(centrality, entity_id), "community": self._lookup(communities, entity_id)},
            "investigative_lead": lead.iloc[0].to_dict() if not lead.empty else {},
        }

    @staticmethod
    def _empty(case_id: str, notice: str) -> dict[str, Any]:
        return {"type": "FeatureCollection", "features": [], "case_id": case_id, "unmapped_locations": [], "notice": notice}
