## Active TRACIA workflow

Run `python main.py` to process only newly uploaded case documents. Files are
identified by SHA-256, so a case already recorded in the upload manifest is not
extracted again. The active workflow is:

`Upload → document processing → case understanding → dynamic extraction schema → entity/evidence extraction → tabular cases/entities/evidence → relationship extraction → entity resolution → Neo4j graph → graph analytics → FP-Growth/association rules/role mining → Spearman/Bayesian analysis → Lead Match Score → RAG`

Neo4j is built from resolved tabular rows, not from raw document extraction.
The older bulk CSV ingestion and preprocessing modules remain in the repository
only as unused legacy reference code; they are not executed by `main.py`.

## Chroma evidence storage

The RAG stage stores every evidence chunk, embedding, and source metadata in
the durable Chroma collection `tracia_case_evidence`; it no longer creates or
loads a FAISS index in application memory. Every index run replaces this
collection, preventing evidence from an earlier case batch leaking into the
current investigation.

Local persistence (`artifacts/chroma`) is the default. To use Chroma Cloud,
put a real `CHROMA_API_KEY` in `.env` and set `CHROMA_MODE=cloud`; the tenant
and database values come from `CHROMA_TENANT` and `CHROMA_DATABASE`.

## Frontend development API

Run the API in reload mode with `uvicorn api.server:app --reload`. The frontend
posts a question to `POST /api/copilot/ask`; the response includes `answer`,
`interaction_id`, and `chroma_record_url`. The exact question and answer are
then retained in the `tracia_agent_interactions` Chroma collection.

Use these read-only endpoints to render the persisted output in a chat/history
panel without invoking the model again:

- `GET /api/chroma/interactions?case_id=<optional>&limit=20`
- `GET /api/chroma/interactions/{interaction_id}`
- `GET /api/chroma/evidence?limit=20`

`GET /api/copilot/history` is an alias for the list endpoint. CORS origins are
controlled with `CORS_ALLOW_ORIGINS` in `.env`.

## Mock case development mode

`data/cases/development/mock_case_northstar.txt` is fictional training data.
With `TRACIA_ENV=development`, the RAG/API use the isolated
`tracia_dev_case_evidence` and `tracia_dev_agent_interactions` Chroma
collections rather than production collections or incoming case files.

Seed the development collection with:

```powershell
python scripts/seed_development_chroma.py
```

Then start `uvicorn api.server:app --reload` and ask the Copilot endpoint.

### Upload-to-output frontend flow

In development mode, upload a supported case file (`.txt`, `.csv`, `.json`,
`.pdf`, `.png`, `.jpg`, `.jpeg`) to `POST /api/development/cases/upload` as
multipart field `file`. The endpoint returns the generated document records,
case profile, extraction schema, entities, evidence, relationships, completed
stages, and `case_id` in one response. It chunks the development documents and
writes them to Chroma before responding.

For the uploaded case, use `POST /api/copilot/ask` with its `case_id`. In
development mode this uses Chroma evidence but does not read or change the
shared Neo4j graph. Render indexed chunks with `GET /api/chroma/evidence` and
stored answers with `GET /api/chroma/interactions`.

## Spatial-intelligence API

The map integration is intentionally an evidence-grounded, read-only API. It
adapts the tabular evidence, resolved graph relationships, and existing
analytics into GeoJSON for a map client; it does not geocode, infer movement,
or invent spatial links.

Run it locally with:

```powershell
uvicorn api.server:app --reload
```

Run the API in Docker with:

```powershell
docker build -t criminal-network-api .
docker run --rm -p 8000:8000 --env-file .env criminal-network-api
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
