# MedX — Medical Content Recommender

A hybrid recommender system prototype built to demonstrate core ML engineering skills
for the **coliquio Data Scientist / ML Engineer** role.

## What it does

Recommends relevant medical articles to doctors on a community health platform,
using a **hybrid approach** combining:

| Method | Technique | Library |
|---|---|---|
| Content-based filtering | TF-IDF + cosine similarity | scikit-learn |
| Collaborative filtering | SVD matrix factorization | scikit-surprise |

## Tech Stack

- **API**: FastAPI + Uvicorn
- **ML**: scikit-learn, scikit-surprise, Pandas, NumPy
- **Frontend**: Vanilla HTML/CSS/JS (no build step)
- **Deploy**: Render

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

## Deploy to Render

1. Push to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Connect your repo — Render auto-detects `render.yaml`
4. Deploy 🚀
