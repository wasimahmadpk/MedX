# MedX

> The right medical article, for the right doctor, at the right time.

MedX is a hybrid medical content recommender — a working prototype that personalises articles for doctors using specialty, reading behaviour, and **time of day**.

---

## Features

- **Hybrid recommendations** — content-based (TF-IDF) + collaborative filtering (SVD)
- **Context-aware ranking** — boosts short, simple reads at lunch; deeper content in the morning or evening
- **Live tuning** — blend slider (α) to shift between content-based and collaborative signals
- **Article modal** — click any article for summary, read time, complexity, and similar items
- **Up to 4 recommendations** per request — focused, not overwhelming

---

## How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────────┐
│ Content-based   │     │ Collaborative   │     │ Time-context         │
│ TF-IDF + cosine │  +  │ SVD factorisation│ ──► │ re-ranking by hour   │
│ (specialty,     │     │ (reading history)│     │ (complexity + length)│
│  tags, history) │     │                 │     │                      │
└─────────────────┘     └─────────────────┘     └──────────────────────┘
         │                       │                          │
         └─────────── α blend ──┘                          │
                              hybrid score ────────────────►│
                                              final ranking │
```

| Layer | Method | Library |
|---|---|---|
| Content-based | TF-IDF + cosine similarity | scikit-learn |
| Collaborative | Mean-centred SVD (R ≈ U·Σ·Vᵀ) | NumPy |
| Time-context | Rule-based re-ranking by hour | NumPy |

**Hybrid score:** `α · content + (1 − α) · collaborative`  
**Final score:** `hybrid × context_boost` (complexity + reading time vs time slot)

### Time slots

| Period | Hours | Prefers |
|---|---|---|
| Early Morning | 05–09 | Long, complex reads |
| Morning Work | 09–12 | Medium length |
| Lunch Break | 12–14 | Short (≤5 min), simple |
| Afternoon Work | 14–18 | Medium |
| Evening | 18–22 | Long, complex |
| Late Night | 22–05 | Short reads |

The UI sends the browser’s local hour automatically; the API accepts `?hour=0–23`.

---

## Project Structure

```
MedX/
├── main.py              # FastAPI API + embedded frontend
├── recommender/
│   └── engine.py        # Hybrid engine + context re-ranker
├── data/
│   └── seed_data.py     # Doctors, articles, interactions
├── vercel.json          # Vercel deployment
└── requirements.txt
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Uvicorn |
| ML | scikit-learn, NumPy, Pandas |
| Frontend | HTML, CSS, JavaScript (embedded in `main.py`) |
| Deploy | Vercel (Python serverless) |

---

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web UI |
| GET | `/api/doctors` | List doctors |
| GET | `/api/doctors/{id}` | Profile + reading history |
| GET | `/api/recommend/{id}` | Personalised recs (`?n=4&alpha=0.5&hour=12`, max 4) |
| GET | `/api/articles` | All articles |
| GET | `/api/articles/{id}/similar` | Similar articles (content-based) |
| GET | `/api/health` | Health check |
| GET | `/docs` | Swagger UI |

---

## Run Locally

```bash
git clone https://github.com/wasimahmadpk/MedX.git
cd MedX
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000).

---

## Deploy

Connect the repo at [vercel.com/new](https://vercel.com/new) — Vercel reads `vercel.json` automatically.

```bash
npm i -g vercel
vercel --prod
```

---

## Dataset (synthetic)

| | Count |
|---|---:|
| Doctors | 15 (8 specialties) |
| Articles | 40 (guidelines, reviews, education, quick lunch reads) |
| Interactions | 90+ ratings (1–5) |

Each article includes `complexity_score` (0–1) and `reading_time_minutes`. Fourteen articles are tagged for **lunch-break** reading (≤5 min, low complexity).

---

## License

MIT — use freely for learning and portfolio demos.
