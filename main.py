"""
MedX — Medical Content Recommender API
Prototype demonstrating hybrid recommender systems for the coliquio doctor platform.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from recommender.engine import MedXRecommender

PUBLIC_DIR = Path(__file__).parent / "public"

app = FastAPI(
    title="MedX Recommender",
    description="Hybrid ML recommender system for medical content — coliquio prototype",
    version="0.1.0",
)

# Lazy singleton — initialized on first request, reused within a warm Vercel instance
_recommender: MedXRecommender | None = None


def get_rec() -> MedXRecommender:
    global _recommender
    if _recommender is None:
        _recommender = MedXRecommender()
    return _recommender


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(PUBLIC_DIR / "index.html")


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/doctors", tags=["Doctors"])
async def list_doctors():
    return get_rec().get_all_doctors()


@app.get("/api/doctors/{doctor_id}", tags=["Doctors"])
async def get_doctor(doctor_id: str):
    doc = get_rec().get_doctor(doctor_id)
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

    - **alpha**: content-based weight 0–1 (0 = pure collaborative, 1 = pure content)
    """
    doc = get_rec().get_doctor(doctor_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Doctor not found")
    recs = get_rec().recommend(doctor_id, n=n, alpha=alpha, exclude_read=exclude_read)
    return {"doctor": doc, "recommendations": recs}


@app.get("/api/articles", tags=["Articles"])
async def list_articles():
    return get_rec().get_all_articles()


@app.get("/api/articles/{article_id}", tags=["Articles"])
async def get_article(article_id: str):
    art = get_rec().get_article(article_id)
    if not art:
        raise HTTPException(status_code=404, detail="Article not found")
    return art


@app.get("/api/articles/{article_id}/similar", tags=["Articles"])
async def similar_articles(article_id: str, n: int = 4):
    art = get_rec().get_article(article_id)
    if not art:
        raise HTTPException(status_code=404, detail="Article not found")
    return {"article": art, "similar": get_rec().similar_articles(article_id, n=n)}


@app.get("/api/health", tags=["System"])
async def health():
    return {"status": "ok", "model": "hybrid (TF-IDF + numpy SVD)", "version": "0.1.0"}
