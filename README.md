## Active TRACIA workflow

Run `python main.py` to process only newly uploaded case documents. Files are
identified by SHA-256, so a case already recorded in the upload manifest is not
extracted again. The active workflow is:

`Upload → document processing → case understanding → dynamic extraction schema → entity/evidence extraction → tabular cases/entities/evidence → relationship extraction → entity resolution → Neo4j graph → graph analytics → FP-Growth/association rules/role mining → Spearman/Bayesian analysis → Lead Match Score → RAG`

Neo4j is built from resolved tabular rows, not from raw document extraction.
The older bulk CSV ingestion and preprocessing modules remain in the repository
only as unused legacy reference code; they are not executed by `main.py`.

## Spatial-intelligence API

The map integration is intentionally an evidence-grounded, read-only API. It
adapts the tabular evidence, resolved graph relationships, and existing
analytics into GeoJSON for a map client; it does not geocode, infer movement,
or invent spatial links.

Run it locally with:

```powershell
uvicorn api.server:app --reload
```

Available endpoints:

- `GET /api/v1/map/cases` lists cases that have graph relationship output.
- `GET /api/v1/map/cases/{case_id}` returns a GeoJSON `FeatureCollection` plus
  `unmapped_locations`. Optional filters: `entity_id`, `entity_type`,
  `date_from`, and `date_to` (`YYYY-MM-DD`).
- `GET /api/v1/map/entities/{entity_id}/locations?case_id=...` supports a
  graph-to-map entity selection.

When a documented location does not have stored coordinates it is returned as
text in `unmapped_locations`, with evidence metadata, rather than displayed at
an invented point. The current repository has no frontend map component or map
SDK, so no provider or duplicate UI has been introduced.
