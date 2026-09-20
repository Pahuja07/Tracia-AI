"""Read-only API routes consumed by the existing/future investigator map."""
from fastapi import APIRouter, HTTPException, Query

from src.criminalNetwork.components.spatial_intelligence import SpatialIntelligenceService
from src.criminalNetwork.config.configuration import ConfigurationManager

router = APIRouter(prefix="/api/v1", tags=["spatial-intelligence"])


def _service() -> SpatialIntelligenceService:
    return SpatialIntelligenceService(ConfigurationManager().get_spatial_intelligence_config())


@router.get("/map/cases")
def list_map_cases():
    return {"cases": _service().case_ids()}


@router.get("/map/cases/{case_id}")
def get_case_map_data(
    case_id: str,
    entity_id: str | None = None,
    entity_type: str | None = None,
    date_from: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    date_to: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
):
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from must not be after date_to")
    return _service().map_data(case_id, entity_id, entity_type, date_from, date_to)


@router.get("/map/entities/{entity_id}/locations")
def get_entity_locations(entity_id: str, case_id: str = Query(...)):
    return _service().map_data(case_id, entity_id=entity_id)
