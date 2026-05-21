"""
MedX — Medical Content Recommender API
Prototype demonstrating hybrid recommender systems for the coliquio doctor platform.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from recommender.engine import MedXRecommender

# Embed the frontend HTML inline so Vercel's Python builder packages it correctly.
# (Vercel only bundles .py files — external file references like FileResponse break.)
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MedX</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --blue:      #1a6fbf;
      --blue-light:#e8f2fc;
      --blue-mid:  #2d88d4;
      --green:     #19a974;
      --gray-50:   #f8fafc;
      --gray-100:  #f1f5f9;
      --gray-200:  #e2e8f0;
      --gray-400:  #94a3b8;
      --gray-600:  #475569;
      --gray-800:  #1e293b;
      --white:     #ffffff;
      --radius:    10px;
      --shadow:    0 2px 12px rgba(0,0,0,.08);
    }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--gray-50); color: var(--gray-800); min-height: 100vh; }
    header { background: var(--blue); color: white; padding: 0 2rem; display: flex; align-items: center; justify-content: space-between; height: 60px; box-shadow: 0 2px 8px rgba(0,0,0,.15); position: sticky; top: 0; z-index: 100; }
    .logo { font-size: 1.35rem; font-weight: 700; letter-spacing: -.5px; }
    .logo span { color: #7dd3fc; }
    .badge { font-size: .7rem; background: rgba(255,255,255,.2); padding: 3px 8px; border-radius: 20px; letter-spacing: .5px; }
    .layout { display: grid; grid-template-columns: 300px 1fr; gap: 1.5rem; max-width: 1200px; margin: 1.5rem auto; padding: 0 1.5rem; }
    .sidebar { display: flex; flex-direction: column; gap: 1rem; }
    .card { background: var(--white); border-radius: var(--radius); box-shadow: var(--shadow); overflow: hidden; }
    .card-header { background: var(--blue); color: white; padding: .65rem 1rem; font-size: .8rem; font-weight: 600; text-transform: uppercase; letter-spacing: .8px; }
    .card-body { padding: 1rem; }
    select { width: 100%; padding: .55rem .75rem; border: 1.5px solid var(--gray-200); border-radius: 6px; font-size: .9rem; background: white; color: var(--gray-800); cursor: pointer; outline: none; transition: border-color .2s; }
    select:focus { border-color: var(--blue); }
    .doctor-profile { display: none; }
    .doctor-avatar { width: 52px; height: 52px; background: var(--blue-light); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; margin-bottom: .75rem; }
    .doctor-name { font-weight: 700; font-size: 1rem; margin-bottom: .2rem; }
    .doctor-specialty { display: inline-block; background: var(--blue-light); color: var(--blue); font-size: .72rem; font-weight: 600; padding: 2px 8px; border-radius: 20px; text-transform: capitalize; margin-bottom: .6rem; }
    .doctor-meta { font-size: .8rem; color: var(--gray-600); }
    .doctor-meta span { display: block; margin-top: .2rem; }
    .slider-group { display: flex; flex-direction: column; gap: .4rem; }
    .slider-label { display: flex; justify-content: space-between; font-size: .8rem; color: var(--gray-600); }
    input[type=range] { width: 100%; accent-color: var(--blue); cursor: pointer; }
    .slider-hints { display: flex; justify-content: space-between; font-size: .68rem; color: var(--gray-400); }
    .tabs { display: flex; border-bottom: 2px solid var(--gray-200); margin-bottom: 1rem; }
    .tab-btn { padding: .55rem 1.1rem; border: none; background: none; font-size: .85rem; font-weight: 500; color: var(--gray-600); cursor: pointer; border-bottom: 2px solid transparent; margin-bottom: -2px; transition: all .2s; }
    .tab-btn.active { color: var(--blue); border-bottom-color: var(--blue); font-weight: 700; }
    .tab-btn:hover:not(.active) { color: var(--gray-800); }
    .main { display: flex; flex-direction: column; gap: 1.5rem; }
    .empty-state { text-align: center; padding: 4rem 2rem; color: var(--gray-400); }
    .empty-state .icon { font-size: 3rem; margin-bottom: 1rem; }
    .empty-state h3 { font-size: 1.1rem; color: var(--gray-600); margin-bottom: .5rem; }
    .empty-state p { font-size: .85rem; }
    .article-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1rem; }
    .article-card { background: white; border-radius: var(--radius); box-shadow: var(--shadow); padding: 1rem; border-left: 4px solid var(--blue); transition: transform .15s, box-shadow .15s; cursor: pointer; }
    .article-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,.12); }
    .article-card.selected { border-left-color: var(--green); box-shadow: 0 0 0 2px var(--green); }
    .art-type { font-size: .65rem; font-weight: 700; text-transform: uppercase; letter-spacing: .8px; color: var(--gray-400); margin-bottom: .3rem; }
    .art-title { font-weight: 600; font-size: .9rem; line-height: 1.35; margin-bottom: .5rem; }
    .art-summary { font-size: .78rem; color: var(--gray-600); line-height: 1.5; margin-bottom: .65rem; }
    .tags { display: flex; flex-wrap: wrap; gap: .3rem; }
    .tag { font-size: .65rem; background: var(--gray-100); color: var(--gray-600); padding: 2px 7px; border-radius: 20px; }
    .tag.specialty-tag { background: var(--blue-light); color: var(--blue); font-weight: 600; }
    .art-score { margin-top: .65rem; padding-top: .65rem; border-top: 1px solid var(--gray-100); display: flex; align-items: center; gap: .5rem; font-size: .75rem; color: var(--gray-600); }
    .score-bar { flex: 1; height: 4px; background: var(--gray-200); border-radius: 4px; overflow: hidden; }
    .score-fill { height: 100%; background: var(--blue); border-radius: 4px; }
    .similar-panel { background: white; border-radius: var(--radius); box-shadow: var(--shadow); overflow: hidden; }
    .similar-panel .panel-header { background: var(--gray-100); padding: .7rem 1rem; font-size: .8rem; font-weight: 600; color: var(--gray-600); border-bottom: 1px solid var(--gray-200); }
    .similar-list { padding: .5rem; }
    .similar-item { display: flex; align-items: flex-start; gap: .75rem; padding: .6rem .5rem; border-radius: 6px; cursor: pointer; transition: background .15s; }
    .similar-item:hover { background: var(--gray-50); }
    .sim-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--blue-mid); margin-top: 5px; flex-shrink: 0; }
    .sim-title { font-size: .82rem; font-weight: 500; line-height: 1.35; }
    .sim-meta { font-size: .7rem; color: var(--gray-400); margin-top: .15rem; }
    .spinner { display: none; width: 24px; height: 24px; border: 3px solid var(--gray-200); border-top-color: var(--blue); border-radius: 50%; animation: spin .7s linear infinite; margin: 2rem auto; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .section-title { font-size: .75rem; font-weight: 700; text-transform: uppercase; letter-spacing: .8px; color: var(--gray-400); margin-bottom: .75rem; }
    .read-list { display: flex; flex-direction: column; gap: .5rem; }
    .read-item { font-size: .78rem; padding: .4rem .6rem; background: var(--gray-50); border-radius: 6px; border-left: 3px solid var(--gray-200); color: var(--gray-600); }
    @media (max-width: 768px) { .layout { grid-template-columns: 1fr; } .article-grid { grid-template-columns: 1fr; } }
    .btn { display: inline-flex; align-items: center; gap: .4rem; padding: .5rem 1rem; background: var(--blue); color: white; border: none; border-radius: 6px; font-size: .85rem; font-weight: 600; cursor: pointer; transition: background .2s; }
    .btn:hover { background: var(--blue-mid); }
    .btn:disabled { background: var(--gray-400); cursor: not-allowed; }
  </style>
</head>
<body>
<header>
  <div class="logo">Med<span>X</span></div>
</header>
<div class="layout">
  <aside class="sidebar">
    <div class="card">
      <div class="card-header">Select Doctor</div>
      <div class="card-body" style="display:flex;flex-direction:column;gap:.75rem;">
        <select id="doctorSelect" onchange="onDoctorChange()">
          <option value="">— Choose a doctor —</option>
        </select>
        <button class="btn" id="getRecsBtn" disabled onclick="fetchRecommendations()">Get Recommendations</button>
      </div>
    </div>
    <div class="card doctor-profile" id="doctorProfile">
      <div class="card-header">Doctor Profile</div>
      <div class="card-body">
        <div class="doctor-avatar">👨‍⚕️</div>
        <div class="doctor-name" id="profName">—</div>
        <div class="doctor-specialty" id="profSpecialty">—</div>
        <div class="doctor-meta">
          <span id="profExp">—</span>
          <span id="profRead">—</span>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="card-header">Algorithm Controls</div>
      <div class="card-body">
        <div class="slider-group">
          <div class="slider-label">
            <span>Recommendation blend</span>
            <strong id="alphaVal">0.5</strong>
          </div>
          <input type="range" id="alphaSlider" min="0" max="1" step="0.1" value="0.5" oninput="document.getElementById('alphaVal').textContent=this.value" />
          <div class="slider-hints"><span>More collaborative</span><span>More content-based</span></div>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="card-header">About this prototype</div>
      <div class="card-body" style="font-size:.78rem;color:var(--gray-600);line-height:1.6;">
        <p><strong>Hybrid recommender</strong> combining:</p>
        <ul style="margin:.5rem 0 .5rem 1rem;">
          <li><strong>TF-IDF + cosine similarity</strong> (content-based)</li>
          <li><strong>SVD matrix factorization</strong> (collaborative, numpy)</li>
        </ul>
        <p>Built with <strong>FastAPI · scikit-learn · NumPy · Pandas</strong></p>
      </div>
    </div>
  </aside>
  <main class="main">
    <div class="card" id="mainCard">
      <div class="card-header"><span id="mainTitle">Recommendations</span></div>
      <div class="card-body">
        <div class="tabs">
          <button class="tab-btn active" id="tabRec" onclick="switchTab('rec')">Recommended</button>
          <button class="tab-btn" id="tabAll" onclick="switchTab('all')">All Articles</button>
          <button class="tab-btn" id="tabHistory" onclick="switchTab('history')">Reading History</button>
        </div>
        <div class="spinner" id="spinner"></div>
        <div class="empty-state" id="emptyState">
          <div class="icon">🔬</div>
          <h3>Select a doctor to get started</h3>
          <p>Choose a doctor from the sidebar and click<br><strong>Get Recommendations</strong></p>
        </div>
        <div id="recPanel" style="display:none;">
          <div class="section-title" id="recSubtitle"></div>
          <div class="article-grid" id="recGrid"></div>
        </div>
        <div id="allPanel" style="display:none;">
          <div class="article-grid" id="allGrid"></div>
        </div>
        <div id="historyPanel" style="display:none;">
          <div class="read-list" id="readList"></div>
        </div>
      </div>
    </div>
    <div class="similar-panel" id="similarPanel" style="display:none;">
      <div class="panel-header" id="similarTitle">Similar Articles</div>
      <div class="similar-list" id="similarList"></div>
    </div>
  </main>
</div>
<script>
const API = '';
let allArticles = [];
let currentDoctor = null;
let currentTab = 'rec';

async function init() {
  const res = await fetch(`${API}/api/doctors`);
  const doctors = await res.json();
  const sel = document.getElementById('doctorSelect');
  doctors.forEach(d => {
    const opt = document.createElement('option');
    opt.value = d.id;
    opt.textContent = `${d.name} · ${d.specialty.replace(/_/g,' ')}`;
    sel.appendChild(opt);
  });
  const artRes = await fetch(`${API}/api/articles`);
  allArticles = await artRes.json();
  renderAllArticles();
}

async function onDoctorChange() {
  const id = document.getElementById('doctorSelect').value;
  if (!id) { document.getElementById('getRecsBtn').disabled = true; document.getElementById('doctorProfile').style.display = 'none'; return; }
  document.getElementById('getRecsBtn').disabled = false;
  const res = await fetch(`${API}/api/doctors/${id}`);
  currentDoctor = await res.json();
  document.getElementById('doctorProfile').style.display = 'block';
  document.getElementById('profName').textContent = currentDoctor.name;
  document.getElementById('profSpecialty').textContent = currentDoctor.specialty.replace(/_/g, ' ');
  document.getElementById('profExp').textContent = `${currentDoctor.years_exp} years of experience`;
  document.getElementById('profRead').textContent = `${currentDoctor.articles_read.length} articles read`;
  renderHistory();
}

async function fetchRecommendations() {
  const id = document.getElementById('doctorSelect').value;
  if (!id) return;
  const alpha = document.getElementById('alphaSlider').value;
  setLoading(true);
  switchTab('rec');
  const res = await fetch(`${API}/api/recommend/${id}?n=6&alpha=${alpha}`);
  const data = await res.json();
  setLoading(false);
  document.getElementById('emptyState').style.display = 'none';
  document.getElementById('recPanel').style.display = 'block';
  document.getElementById('recSubtitle').textContent = `Top recommendations for ${data.doctor.name} · blend α=${alpha}`;
  const grid = document.getElementById('recGrid');
  grid.innerHTML = '';
  data.recommendations.forEach(art => grid.appendChild(buildArticleCard(art, true)));
}

function switchTab(tab) {
  currentTab = tab;
  ['rec','all','history'].forEach(t => {
    document.getElementById(`tab${t.charAt(0).toUpperCase()+t.slice(1)}`).classList.remove('active');
    const panel = document.getElementById(`${t}Panel`);
    if (panel) panel.style.display = 'none';
  });
  document.getElementById(`tab${tab.charAt(0).toUpperCase()+tab.slice(1)}`).classList.add('active');
  if (tab === 'rec') {
    const hasRecs = document.getElementById('recGrid').children.length > 0;
    document.getElementById('recPanel').style.display = hasRecs ? 'block' : 'none';
    document.getElementById('emptyState').style.display = hasRecs ? 'none' : 'block';
  } else if (tab === 'all') {
    document.getElementById('allPanel').style.display = 'block';
    document.getElementById('emptyState').style.display = 'none';
  } else if (tab === 'history') {
    document.getElementById('historyPanel').style.display = 'block';
    document.getElementById('emptyState').style.display = 'none';
    renderHistory();
  }
}

function buildArticleCard(art, showScore = false) {
  const card = document.createElement('div');
  card.className = 'article-card';
  card.dataset.id = art.id;
  const scoreVal = art.score ?? art.similarity ?? 0;
  const scorePct = Math.round(scoreVal * 100);
  card.innerHTML = `
    <div class="art-type">${art.type.replace(/_/g,' ')}</div>
    <div class="art-title">${art.title}</div>
    <div class="art-summary">${art.summary}</div>
    <div class="tags">
      <span class="tag specialty-tag">${art.specialty.replace(/_/g,' ')}</span>
      ${art.tags.slice(0,3).map(t => `<span class="tag">${t}</span>`).join('')}
    </div>
    ${showScore ? `<div class="art-score"><span>Relevance</span><div class="score-bar"><div class="score-fill" style="width:${scorePct}%"></div></div><span>${scorePct}%</span></div>` : ''}
  `;
  card.onclick = () => loadSimilar(art.id, art.title);
  return card;
}

function renderAllArticles() {
  const grid = document.getElementById('allGrid');
  grid.innerHTML = '';
  allArticles.forEach(art => grid.appendChild(buildArticleCard(art, false)));
}

function renderHistory() {
  const list = document.getElementById('readList');
  list.innerHTML = '';
  if (!currentDoctor || !currentDoctor.articles_read.length) {
    list.innerHTML = '<p style="font-size:.82rem;color:var(--gray-400);padding:.5rem">No reading history yet.</p>';
    return;
  }
  currentDoctor.articles_read.forEach(aid => {
    const art = allArticles.find(a => a.id === aid);
    if (!art) return;
    const item = document.createElement('div');
    item.className = 'read-item';
    item.innerHTML = `<strong>${art.title}</strong> <span style="opacity:.6">· ${art.specialty.replace(/_/g,' ')}</span>`;
    list.appendChild(item);
  });
}

async function loadSimilar(articleId, title) {
  document.querySelectorAll('.article-card').forEach(c => c.classList.remove('selected'));
  document.querySelectorAll(`[data-id="${articleId}"]`).forEach(c => c.classList.add('selected'));
  const res = await fetch(`${API}/api/articles/${articleId}/similar?n=4`);
  const data = await res.json();
  document.getElementById('similarTitle').textContent = `Similar to: "${title}"`;
  const list = document.getElementById('similarList');
  list.innerHTML = '';
  data.similar.forEach(art => {
    const item = document.createElement('div');
    item.className = 'similar-item';
    item.innerHTML = `<div class="sim-dot"></div><div><div class="sim-title">${art.title}</div><div class="sim-meta">${art.specialty.replace(/_/g,' ')} · ${art.type.replace(/_/g,' ')} · ${Math.round(art.similarity*100)}% similar</div></div>`;
    item.onclick = () => loadSimilar(art.id, art.title);
    list.appendChild(item);
  });
  document.getElementById('similarPanel').style.display = 'block';
}

function setLoading(on) {
  document.getElementById('spinner').style.display = on ? 'block' : 'none';
  document.getElementById('recPanel').style.display = 'none';
  document.getElementById('emptyState').style.display = 'none';
}

init();
</script>
</body>
</html>"""

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
    return HTMLResponse(_HTML)


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
