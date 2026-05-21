# MedX — Medical Content Recommender

A hybrid recommender system prototype built to demonstrate core ML engineering skills
for the **coliquio Data Scientist / ML Engineer** role.

## What it does

Recommends relevant medical articles to doctors on a community health platform,
using a **hybrid approach** combining:

| Method | Technique | Library |
|---|---|---|
| Content-based filtering | TF-IDF + cosine similarity | scikit-learn |
| Collaborative filtering | SVD matrix factorization | numpy (pure, no extra deps) |

## Tech Stack

- **API**: FastAPI + Uvicorn
- **ML**: scikit-learn, Pandas, NumPy (SVD via numpy.linalg)
- **Frontend**: Vanilla HTML/CSS/JS served via Vercel CDN
- **Deploy**: Vercel

## Run locally

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn main:app --reload

# 4. Open http://localhost:8000
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/doctors` | List all doctors |
| `GET /api/doctors/{id}` | Doctor profile + reading history |
| `GET /api/recommend/{id}?alpha=0.5` | Personalised recommendations |
| `GET /api/articles` | All medical articles |
| `GET /api/articles/{id}/similar` | Content-based similar articles |
| `GET /api/health` | Health check |

Interactive API docs: `http://localhost:8000/docs`

## Deploy to Vercel

```bash
npm i -g vercel
vercel --prod
```

Or connect the GitHub repo at [vercel.com/new](https://vercel.com/new) — Vercel auto-detects `vercel.json`.
