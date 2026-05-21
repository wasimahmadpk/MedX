# MedX — Medical Content Recommender

> A hybrid recommender system prototype simulating the core ML challenge at **coliquio** — recommending relevant medical content to 190,000+ doctors on Germany's largest doctor community platform.

---

## What It Does

Doctors on coliquio receive personalised article recommendations based on their **medical specialty** and **reading behaviour**. MedX demonstrates this using a two-signal hybrid approach:

| Signal | Method | Library |
|---|---|---|
| **Content-based** | TF-IDF vectorisation of article tags + cosine similarity | scikit-learn |
| **Collaborative** | Mean-centred SVD matrix factorisation (R ≈ U·Σ·Vᵀ) | numpy |

The two scores are normalised and blended via a tunable **α parameter**, allowing the system to lean more content-heavy or collaboration-heavy at inference time.

---

## Architecture

```
MedX/
├── main.py                  # FastAPI app — all /api/* routes
├── recommender/
│   └── engine.py            # Hybrid recommender (TF-IDF + numpy SVD)
├── data/
│   └── seed_data.py         # Synthetic doctors, articles & interactions
├── public/
│   └── index.html           # Frontend UI (served by Vercel CDN)
├── vercel.json              # Deployment config — routes API to Python, static to CDN
└── requirements.txt
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| ML | scikit-learn, NumPy, Pandas |
| Recommender | TF-IDF (content) + truncated SVD via `numpy.linalg` (collaborative) |
| Frontend | Vanilla HTML / CSS / JS — no build step |
| Deployment | Vercel (Python serverless + CDN for static) |

---

## How the Recommender Works

### 1. Content-Based Filtering
Each article is represented as a TF-IDF vector over its **tags, specialty, and type**. A doctor's profile vector is the mean of their read articles plus a specialty signal. Articles are ranked by cosine similarity to this profile.

### 2. Collaborative Filtering (numpy SVD)
A **user × article** rating matrix is decomposed using truncated SVD:

```
R ≈ U · Σ · Vᵀ
```

Missing ratings are predicted from the low-rank approximation. Per-user mean centering removes rating bias before decomposition.

### 3. Hybrid Blending
Both score vectors are min-max normalised and combined:

```
score = α · content_score + (1 − α) · collab_score
```

`α = 0.5` by default. The UI exposes a slider to adjust this live.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/doctors` | List all doctors |
| GET | `/api/doctors/{id}` | Doctor profile + reading history |
| GET | `/api/recommend/{id}` | Personalised recommendations (`?n=6&alpha=0.5`) |
| GET | `/api/articles` | All medical articles |
| GET | `/api/articles/{id}/similar` | Content-based similar articles |
| GET | `/api/health` | Health check + model info |

Interactive Swagger docs available at `/docs`.

---

## Run Locally

```bash
# 1. Clone and set up environment
git clone https://github.com/wasimahmadpk/MedX.git
cd MedX
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn main:app --reload

# 4. Open in browser
open http://localhost:8000
```

---

## Deploy to Vercel

**Option A — Vercel CLI:**
```bash
npm i -g vercel
vercel --prod
```

**Option B — GitHub integration (recommended):**
1. Go to [vercel.com/new](https://vercel.com/new)
2. Import `wasimahmadpk/MedX`
3. Click **Deploy** — Vercel auto-reads `vercel.json`

Every `git push` to `main` triggers an automatic redeploy.

---

## Data

The prototype uses synthetic data representing the coliquio platform:

- **15 doctors** across 8 specialties (cardiology, neurology, oncology, etc.)
- **22 medical articles** covering guidelines, reviews, clinical evidence and education
- **60+ doctor–article interactions** with ratings (1–5) forming the collaborative signal

---

## Context

Built as a portfolio prototype to demonstrate the core technical skills required for the **Data Scientist / ML Engineer** role at [coliquio](https://www.coliquio.de) — specifically:

- Recommender system design (collaborative + content-based)
- Python ML stack (scikit-learn, NumPy, Pandas)
- End-to-end delivery from PoC to deployed product
