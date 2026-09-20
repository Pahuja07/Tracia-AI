"""ASGI entry point. Run with: uvicorn api.server:app --reload"""
from fastapi import FastAPI

from api.routes import router

app = FastAPI(title="TRACIA Spatial Intelligence API", version="1.0.0")
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
