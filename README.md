# MedX — Medical Content Recommender

Doctors are overwhelmed with medical literature. MedX cuts through the noise — surfacing the right article, for the right doctor, at the right time of day.

---

## What It Does

MedX recommends personalised medical articles to doctors based on their **specialty**, **reading history**, and **time of day**. It uses a three-layer approach:

| Layer | Method | Library |
|---|---|---|
| **Content-based** | TF-IDF vectorisation + cosine similarity | scikit-learn |
| **Collaborative** | Mean-centred SVD matrix factorisation (R ≈ U·Σ·Vᵀ) | numpy |
| **Time-context** | Complexity + reading time re-ranking by time of day | numpy |

---

## Architecture

```
MedX/
├── main.py                  # FastAPI app — all /api/* routes + embedded frontend
├── recommender/
│   └── engine.py            # Hybrid recommender (TF-IDF + SVD + context re-ranking)
├── data/
│   └── seed_data.py         # Doctors, articles & reading interactions
├── vercel.json              # Vercel deployment config
└── requirements.txt
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| ML | scikit-learn, NumPy, Pandas |
| Frontend | Vanilla HTML / CSS / JS |
| Deployment | Vercel |

---

## How It Works

### 1. Content-Based Filtering
Each article is represented as a TF-IDF vector over its tags, specialty, and type. A doctor's profile is built from their specialty and reading history. Articles are ranked by cosine similarity to this profile.

### 2. Collaborative Filtering (SVD)
A doctor × article rating matrix is decomposed using truncated SVD:

```
R ≈ U · Σ · Vᵀ
```

Predicted ratings fill in the blanks — doctors with similar reading patterns get similar recommendations.

### 3. Hybrid Blending
```
score = α · content_score + (1 − α) · collab_score
```

The α slider in the UI lets you tune the blend live.

### 4. Time-Context Re-Ranking
Each article has a `complexity_score` and `reading_time_minutes`. The system detects the time of day and boosts articles that fit — short simple reads at lunch, deeper content in the morning or evening.

```
final_score = hybrid_score × (1 + 0.3 × context_fit)
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/doctors` | List all doctors |
| GET | `/api/doctors/{id}` | Doctor profile + reading history |
| GET | `/api/recommend/{id}` | Recommendations (`?n=4&alpha=0.5&hour=14`, max 4) |
| GET | `/api/articles` | All articles |
| GET | `/api/articles/{id}/similar` | Similar articles |
| GET | `/api/health` | Health check |

Interactive docs at `/docs`.

---

## Run Locally

```bash
git clone https://github.com/wasimahmadpk/MedX.git
cd MedX
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
# Open http://localhost:8000
```

---

## Deploy to Vercel

```bash
npm i -g vercel
vercel --prod
```

Or connect the repo at [vercel.com/new](https://vercel.com/new) — Vercel auto-reads `vercel.json`.

---

## Data

- **15 doctors** across 8 specialties
- **32 medical articles** — guidelines, reviews, clinical evidence, education
- **75+ reading interactions** with ratings (1–5)
