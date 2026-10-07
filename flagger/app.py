"""Local browser app and JSON API. Inference makes no outbound requests."""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from flagger.predict import StalenessClassifier, ROOT


class AnalyzeRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=25000)
    threshold: float = Field(default=0.5, ge=0.1, le=0.9)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier = StalenessClassifier()
    yield


app = FastAPI(title="AI Answer Staleness Flagger", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model": app.state.classifier.metadata["model_kind"], "offline": True}


@app.get("/api/model")
def model_info():
    return app.state.classifier.metadata


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest):
    if not request.answer.strip():
        raise HTTPException(status_code=400, detail="Paste an answer containing at least one sentence.")
    try:
        result = app.state.classifier.analyze(request.answer, request.threshold)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not result["claims"]:
        raise HTTPException(status_code=400, detail="No sentences found. Headings and code blocks are skipped.")
    return result
