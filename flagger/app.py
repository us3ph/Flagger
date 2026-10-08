"""Local browser app and JSON API. Inference makes no outbound requests."""

from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from flagger.predict import StalenessClassifier, DEFAULT_MODEL, ROOT


class AnalyzeRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=25000)
    threshold: float = Field(default=0.5, ge=0.1, le=0.9)


class ClassifierStore:
    """Pick up completed local model replacements before the next request."""

    def __init__(self, model_path: Path = DEFAULT_MODEL):
        self.model_path = model_path
        self._lock = Lock()
        self._signature = None
        self._classifier = None
        self.get()

    def get(self) -> StalenessClassifier:
        with self._lock:
            try:
                stat = self.model_path.stat()
                signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
                if signature != self._signature:
                    classifier = StalenessClassifier(self.model_path)
                    # Publish a complete instance only after loading succeeds.
                    self._classifier = classifier
                    self._signature = signature
            except Exception as exc:
                raise HTTPException(status_code=503, detail="The local model could not be loaded. Check the model artifact or retrain it.") from exc
            return self._classifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = ClassifierStore()
    yield


app = FastAPI(title="AI Answer Staleness Flagger", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


@app.middleware("http")
async def fresh_responses(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model": app.state.models.get().metadata["model_kind"], "offline": True}


@app.get("/api/model")
def model_info():
    return app.state.models.get().metadata


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest):
    if not request.answer.strip():
        raise HTTPException(status_code=400, detail="Paste an answer containing at least one sentence.")
    try:
        result = app.state.models.get().analyze(request.answer, request.threshold)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not result["claims"]:
        raise HTTPException(status_code=400, detail="No sentences found. Headings and code blocks are skipped.")
    return result
