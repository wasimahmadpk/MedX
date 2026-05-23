# MedX

**Hybrid medical content recommender — personalised articles by specialty, behaviour, and time of day.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Deploy](https://img.shields.io/badge/Deploy-Vercel-000000?logo=vercel&logoColor=white)](https://vercel.com/)

> *The right medical article, for the right doctor, at the right time.*

---

## Overview

Doctors face more medical content than they can read. MedX demonstrates how a recommender can cut through that noise: it ranks articles using **what the doctor specialises in**, **what similar doctors read**, and **when they are likely reading** — a short lunch break vs a quiet evening.

Built as a full-stack prototype: **FastAPI** backend, embedded web UI, deployable on **Vercel**.

---

## Quick start

```bash
git clone https://github.com/wasimahmadpk/MedX.git
cd MedX
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open **http://localhost:8000** → pick a doctor → **Get Recommendations**.

---

## What you can do in the UI

1. **Select a doctor** — 15 profiles across 8 specialties  
2. **Get Recommendations** — up to **4** personalised articles  
3. **Adjust α** — balance content-based vs collaborative signals  
4. **Click an article** — modal with summary, read time, complexity, similar items  
5. **Context banner** — shows current time slot (e.g. Lunch Break) and why items fit  

---

## Recommendation pipeline

```
  Content-based          Collaborative           Time-context
  (TF-IDF + cosine)  +   (SVD)            →    re-ranking
        │                      │                      │
        └──── α · C + (1−α) · S ──── hybrid score ────┘
                                          │
                              final = hybrid × context_boost
```

| Stage | What it uses | Tech |
|---|---|---|
| **Content-based** | Article tags, specialty, type; doctor specialty & history | scikit-learn |
| **Collaborative** | Doctor–article ratings matrix | NumPy SVD |
| **Hybrid blend** | Tunable weight α (default 0.5) | — |
| **Context-aware** | Hour of day, article complexity, reading time | Rule-based boost |

### Context-aware ranking

Recommendations adapt to **time of day** — a simplified version of production context-aware systems:

| Time slot | Hours | Ideal content |
|---|---|---|
| Early Morning | 05–09 | Long, in-depth |
| Morning Work | 09–12 | Medium |
| **Lunch Break** | **12–14** | **Short (≤5 min), low complexity** |
| Afternoon Work | 14–18 | Medium |
| Evening | 18–22 | Long, in-depth |
| Late Night / Night | 22–05 | Short |

The UI passes the browser’s local hour; the API also accepts `?hour=0–23`.

---

## Example API call

```bash
curl "http://localhost:8000/api/recommend/d1?n=4&alpha=0.5&hour=12"
```

Response (abbreviated):

```json
{
  "doctor": { "name": "Dr. Anna Müller", "specialty": "cardiology" },
  "context": {
    "hour": 12,
    "label": "Lunch Break",
    "icon": "🍽️",
    "max_reading_min": 5
  },
  "recommendations": [
    {
      "title": "Vitamin D Deficiency in Primary Care: Test or Treat?",
      "reading_time_minutes": 4,
      "complexity_score": 0.3,
      "score": 0.61
    }
  ]
}
```

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/api/doctors` | List all doctors |
| `GET` | `/api/doctors/{id}` | Profile + reading history |
| `GET` | `/api/recommend/{id}` | Recommendations — `?n=4&alpha=0.5&hour=12` (max **n=4**) |
| `GET` | `/api/articles` | All articles |
| `GET` | `/api/articles/{id}/similar` | Similar articles (TF-IDF) |
| `GET` | `/api/health` | Health check |
| `GET` | `/docs` | Interactive Swagger docs |

---

## Project layout

```
MedX/
├── main.py                 # FastAPI routes + embedded frontend
├── recommender/engine.py   # Hybrid model + context re-ranker
├── data/seed_data.py       # Synthetic doctors, articles, ratings
├── vercel.json
└── requirements.txt
```

---

## Tech stack

| | |
|---|---|
| **Backend** | FastAPI, Uvicorn |
| **ML** | scikit-learn, NumPy, Pandas |
| **Frontend** | HTML, CSS, JavaScript |
| **Deploy** | Vercel serverless Python |

---

## Dataset

Synthetic data modelling a doctor community platform:

| Entity | Details |
|---|---|
| **Doctors** | 15 — cardiology, neurology, oncology, pediatrics, and more |
| **Articles** | 40 — guidelines, reviews, clinical evidence, education |
| **Quick reads** | 14 lunch-friendly articles (≤5 min, low complexity) |
| **Interactions** | 94 doctor–article ratings (1–5) |

Each article has `complexity_score` (0–1) and `reading_time_minutes`.

---

## Deploy to Vercel

1. Fork or import [github.com/wasimahmadpk/MedX](https://github.com/wasimahmadpk/MedX) at [vercel.com/new](https://vercel.com/new)  
2. Deploy — `vercel.json` is included  

Or via CLI:

```bash
npm i -g vercel && vercel --prod
```

---

## Author

**Wasim Ahmad** — [GitHub](https://github.com/wasimahmadpk) · [Portfolio](https://wasimahmadpk.github.io/portfolio/) · [LinkedIn](https://www.linkedin.com/in/wasim-ahmad-73293767)

---

## Related concepts

Hybrid filtering · Matrix factorisation · Context-aware recommendation · Time-aware ranking · Personalisation · A/B experimentation (CUPED, variance reduction)
