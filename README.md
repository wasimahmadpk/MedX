# MedX

> **The right medical article, for the right doctor, at the right time.**

[![Live Demo](https://img.shields.io/badge/demo-live-009688?style=for-the-badge&logo=vercel&logoColor=white)](https://med-x-plum.vercel.app)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ML](https://img.shields.io/badge/ML-scikit--learn%20%7C%20NumPy-F7931E)](https://scikit-learn.org/)

**[Live app](https://med-x-plum.vercel.app)** · **[API docs](https://med-x-plum.vercel.app/docs)** · **[GitHub](https://github.com/wasimahmadpk/MedX)**

MedX is a full-stack hybrid recommender for medical content. It personalises articles using **specialty & tags**, **doctor reading patterns**, and **time of day** — returning up to **5** recommendations via a web UI and REST API.

---

## Contents

- [Why MedX](#why-medx)
- [Live demo](#live-demo)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [API](#api)
- [Dataset](#dataset)
- [FAQ](#faq)
- [Deploy](#deploy)
- [Author](#author)

---

## Why MedX

Doctors are overloaded with literature. A useful recommender must solve three problems at once:

| Signal | Question | Method |
|---|---|---|
| **Relevance** | Does this fit the doctor's specialty? | TF-IDF + cosine similarity |
| **Behaviour** | Do peers with similar tastes read this? | SVD collaborative filtering |
| **Timing** | Can they read it *now*? | Context-aware re-ranking |

**Why hybrid?** Content-only misses cross-specialty discoveries. Collaborative-only fails for new doctors with no history. MedX blends both, then adjusts for the hour.

| Approach | Strength | Weakness |
|---|---|---|
| Content-based only | Works for new users | Misses behavioural patterns |
| Collaborative only | Finds hidden patterns | Cold-start problem |
| **MedX hybrid + context** | Specialty + behaviour + timing | Prototype-scale data |

---

## Live demo

**→ [med-x-plum.vercel.app](https://med-x-plum.vercel.app)**

| Step | Action |
|---|---|
| 1 | Select a doctor from the sidebar |
| 2 | Click **Get Recommendations** |
| 3 | Move the **α slider** (content ↔ collaborative) |
| 4 | Click an article for details & similar items |
| 5 | Check the **context banner** for your time slot |

**Try these doctors**

| ID | Doctor | Specialty |
|---|---|---|
| `d1` | Dr. Anna Müller | Cardiology |
| `d2` | Dr. Ben Schäfer | Neurology |
| `d8` | Dr. Hans Weber | General practice |
| `d10` | Dr. Jonas Schulz | Psychiatry |

---

## How it works

```mermaid
flowchart LR
  A[Doctor profile] --> C[Content score]
  B[Rating matrix] --> S[Collab score]
  C --> H[Hybrid blend]
  S --> H
  T[Hour of day] --> X[Context boost]
  H --> F[Final rank]
  X --> F
  F --> R[Top 5 articles]
```

```
hybrid  = α · content + (1 − α) · collaborative
final   = hybrid × context_boost(complexity, read_time, hour)
```

| Layer | Tech | Purpose |
|---|---|---|
| Content-based | scikit-learn | Match tags, specialty, reading history |
| Collaborative | NumPy SVD | Predict from doctor–article ratings |
| Hybrid blend | α parameter | Balance both signals |
| Context re-rank | Time slots | Boost lunch reads at noon, deep reads at night |

**Time slots:** Early morning & evening → long/complex · **Lunch (12–14)** → ≤5 min, simple · Night → short reads

**Specialties covered:** cardiology · neurology · oncology · pediatrics · dermatology · general practice · psychiatry · radiology

---

## Quick start

```bash
git clone https://github.com/wasimahmadpk/MedX.git
cd MedX
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) · Requires **Python 3.11+**

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/api/recommend/{id}` | Personalised recommendations |
| `GET` | `/api/doctors` | All doctors |
| `GET` | `/api/doctors/{id}` | Profile + reading history |
| `GET` | `/api/articles` | All articles |
| `GET` | `/api/articles/{id}/similar` | Similar articles |
| `GET` | `/api/health` | Health check |
| `GET` | `/docs` | Swagger UI |

**Parameters** — `GET /api/recommend/{id}`

| Param | Default | Description |
|---|---|---|
| `n` | `5` | Max 5 results |
| `alpha` | `0.5` | Content weight (0 = collab, 1 = content) |
| `hour` | auto | 0–23 for context ranking |

```bash
# Cardiologist, lunch break, equal blend
curl "https://med-x-plum.vercel.app/api/recommend/d1?n=5&alpha=0.5&hour=12"

# Same doctor, evening — compare results
curl "https://med-x-plum.vercel.app/api/recommend/d1?n=5&alpha=0.5&hour=20"
```

<details>
<summary>Sample JSON response</summary>

```json
{
  "doctor": { "name": "Dr. Anna Müller", "specialty": "cardiology" },
  "context": { "label": "Lunch Break", "hour": 12, "max_reading_min": 5 },
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

</details>

---

## Dataset

Synthetic demo data in `data/seed_data.py`:

| Entity | Count |
|---|---:|
| Doctors | 15 |
| Articles | 40 |
| Lunch-friendly quick reads | 14 |
| Ratings (1–5) | 94 |

Each article has `complexity_score` (0–1) and `reading_time_minutes`.

---

## FAQ

**What does the α slider do?**  
`α = 1` → pure content-based (specialty & tags). `α = 0` → pure collaborative (reading patterns). Default `0.5` blends both.

**Why only 5 recommendations?**  
Keeps output focused — mimics a curated feed rather than an overwhelming list.

**Is the data real?**  
No — synthetic data for demonstration. The algorithms are production-style; the dataset is not.

**Does context use machine learning?**  
The context layer is rule-based (time slot → ideal complexity & length). A production system would learn these patterns from engagement data.

---

## Project structure

```
MedX/
├── main.py                 # FastAPI + embedded UI
├── recommender/engine.py   # Hybrid engine + context re-ranker
├── data/seed_data.py       # Doctors, articles, interactions
├── vercel.json
└── requirements.txt
```

---

## Deploy

Import at [vercel.com/new](https://vercel.com/new) — `vercel.json` included, no extra config.

```bash
npm i -g vercel && vercel --prod
```

Verify: `curl https://med-x-plum.vercel.app/api/health`

---

## Author

**Wasim Ahmad** — ML Engineer · Data Scientist

[Demo](https://med-x-plum.vercel.app) · [GitHub](https://github.com/wasimahmadpk) · [Portfolio](https://wasimahmadpk.github.io/portfolio/) · [LinkedIn](https://www.linkedin.com/in/wasim-ahmad-73293767)

---

<p align="center">
  <sub>Hybrid filtering · Matrix factorisation · Context-aware recommendation · Time-aware ranking</sub>
</p>
