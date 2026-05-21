"""
MedX — Medical Content Recommender API
Prototype demonstrating hybrid recommender systems for the coliquio doctor platform.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from recommender.engine import MedXRecommender

# ---------------------------------------------------------------------------
# App lifecycle — build the recommender once at startup
# ---------------------------------------------------------------------------
recommender: MedXRecommender | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    recommender = MedXRecommender()
    yield


app = FastAPI(
    title="MedX Recommender",
    description="Hybrid ML recommender system for medical content — coliquio prototype",
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Serve the frontend
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(STATIC_DIR / "index.html")


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/doctors", tags=["Doctors"])
async def list_doctors():
    """List all doctors on the platform."""
    return recommender.get_all_doctors()


@app.get("/api/doctors/{doctor_id}", tags=["Doctors"])
async def get_doctor(doctor_id: str):
    """Get a single doctor's profile and reading history."""
    doc = recommender.get_doctor(doctor_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Doctor not found")
    return doc


@app.get("/api/recommend/{doctor_id}", tags=["Recommendations"])
async def recommend_for_doctor(
    doctor_id: str,
    n: int = 6,
    alpha: float = 0.5,
    exclude_read: bool = True,
):
    """
    Get personalised article recommendations for a doctor.

    - **n**: number of recommendations (default 6)
    - **alpha**: content-based weight 0–1 (0.5 = equal blend)
    - **exclude_read**: hide already-read articles
    """
    doc = recommender.get_doctor(doctor_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Doctor not found")
    recs = recommender.recommend(doctor_id, n=n, alpha=alpha, exclude_read=exclude_read)
    return {"doctor": doc, "recommendations": recs}


@app.get("/api/articles", tags=["Articles"])
async def list_articles():
    """List all available medical articles."""
    return recommender.get_all_articles()


@app.get("/api/articles/{article_id}", tags=["Articles"])
async def get_article(article_id: str):
    """Get a single article."""
    art = recommender.get_article(article_id)
    if not art:
        raise HTTPException(status_code=404, detail="Article not found")
    return art


@app.get("/api/articles/{article_id}/similar", tags=["Articles"])
async def similar_articles(article_id: str, n: int = 4):
    """Get content-based similar articles."""
    art = recommender.get_article(article_id)
    if not art:
        raise HTTPException(status_code=404, detail="Article not found")
    similar = recommender.similar_articles(article_id, n=n)
    return {"article": art, "similar": similar}


@app.get("/api/health", tags=["System"])
async def health():
    return {"status": "ok", "model": "hybrid (TF-IDF + SVD)", "version": "0.1.0"}
