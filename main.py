"""
MedX — Medical Content Recommender API
Prototype demonstrating hybrid recommender systems for the coliquio doctor platform.
"""

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from recommender.engine import MedXRecommender, get_time_slot

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
      --green-light:#e8f7f0;
      --green-dark:#148f63;
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
    header { background: var(--blue); color: white; padding: 0 2rem; display: flex; align-items: center; justify-content: space-between; height: 60px; box-shadow: 0 2px 8px rgba(0,0,0,.15); position: sticky; top: 0; z-index: 100; border-bottom: 3px solid var(--green); }
    .logo { font-size: 1.35rem; font-weight: 700; letter-spacing: -.5px; }
    .logo span { color: var(--green); }
    .header-badge { font-size: .65rem; background: var(--green); color: white; padding: 4px 10px; border-radius: 20px; font-weight: 600; letter-spacing: .4px; }
    .badge { font-size: .7rem; background: rgba(255,255,255,.2); padding: 3px 8px; border-radius: 20px; letter-spacing: .5px; }
    .layout { display: grid; grid-template-columns: 300px 1fr; gap: 1.5rem; max-width: 1200px; margin: 1.5rem auto; padding: 0 1.5rem; }
    .sidebar { display: flex; flex-direction: column; gap: 1rem; }
    .card { background: var(--white); border-radius: var(--radius); box-shadow: var(--shadow); overflow: hidden; }
    .card-header { background: var(--blue); color: white; padding: .65rem 1rem; font-size: .8rem; font-weight: 600; text-transform: uppercase; letter-spacing: .8px; border-left: 4px solid transparent; }
    #mainCard > .card-header { border-left-color: var(--green); }
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
    .main { display: flex; flex-direction: column; gap: 1.5rem; min-width: 0; }
    .empty-state { text-align: center; padding: 4rem 2rem; color: var(--gray-400); }
    .empty-state .icon { font-size: 3rem; margin-bottom: 1rem; }
    .empty-state h3 { font-size: 1.1rem; color: var(--gray-600); margin-bottom: .5rem; }
    .empty-state p { font-size: .85rem; }
    .article-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1rem; }
    .article-card { background: white; border-radius: var(--radius); box-shadow: var(--shadow); padding: 1rem; border-left: 4px solid var(--blue); transition: transform .15s, box-shadow .15s; cursor: pointer; }
    .article-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,.12); }
    .article-card.selected { border-left-color: var(--green); box-shadow: 0 0 0 2px var(--green); }
    .art-type { font-size: .65rem; font-weight: 700; text-transform: uppercase; letter-spacing: .8px; color: var(--gray-400); margin-bottom: .3rem; }
    .art-type.type-highlight { color: var(--green-dark); }
    .type-pill { display: inline-block; background: var(--green-light); color: var(--green-dark); padding: 2px 7px; border-radius: 20px; margin-right: .25rem; }
    .rec-slide:first-child .article-card { border-left-color: var(--green); }
    .rec-rank { display: inline-block; font-size: .62rem; font-weight: 700; background: var(--green); color: white; padding: 2px 7px; border-radius: 20px; margin-bottom: .35rem; letter-spacing: .3px; }
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
    .btn-accent { background: var(--green); }
    .btn-accent:hover:not(:disabled) { background: var(--green-dark); }
    .context-toast { position: fixed; top: 72px; left: 50%; transform: translateX(-50%) translateY(-120%); opacity: 0; z-index: 150; width: min(420px, calc(100vw - 2rem)); pointer-events: none; transition: transform .45s cubic-bezier(.4,0,.2,1), opacity .35s ease; }
    .context-toast.show { transform: translateX(-50%) translateY(0); opacity: 1; pointer-events: auto; }
    .context-toast-inner { background: white; border-radius: 12px; box-shadow: 0 12px 40px rgba(26,111,191,.18), 0 4px 12px rgba(0,0,0,.08); border: 1px solid #bfdbfe; border-left: 4px solid var(--green); padding: .85rem 1rem .85rem .9rem; display: flex; align-items: flex-start; gap: .75rem; position: relative; overflow: hidden; }
    .context-toast-icon { font-size: 1.6rem; line-height: 1; flex-shrink: 0; margin-top: .1rem; }
    .context-toast-body { flex: 1; min-width: 0; }
    .context-toast-title { font-size: .88rem; font-weight: 700; color: var(--blue); margin-bottom: .2rem; }
    .context-toast-msg { font-size: .78rem; color: var(--gray-600); line-height: 1.45; }
    .context-toast-msg strong { color: var(--gray-800); font-weight: 600; }
    .context-toast-close { background: none; border: none; font-size: 1.25rem; line-height: 1; color: var(--gray-400); cursor: pointer; padding: .15rem; margin: -.15rem -.1rem 0 0; flex-shrink: 0; border-radius: 4px; transition: color .15s, background .15s; }
    .context-toast-close:hover { color: var(--gray-800); background: var(--gray-100); }
    .context-toast-progress { height: 3px; background: var(--blue-light); border-radius: 0 0 12px 12px; overflow: hidden; margin-top: -1px; }
    .context-toast-progress span { display: block; height: 100%; background: linear-gradient(90deg, var(--blue), var(--green)); width: 100%; transform-origin: left; animation: toast-progress 5s linear forwards; }
    @keyframes toast-progress { from { transform: scaleX(1); } to { transform: scaleX(0); } }
    .art-meta { display:flex; align-items:center; gap:.75rem; margin-top:.5rem; font-size:.7rem; color:var(--gray-400); }
    .art-meta span { display:flex; align-items:center; gap:.2rem; }
    .complexity-bar { width:40px; height:4px; background:var(--gray-200); border-radius:4px; overflow:hidden; display:inline-block; vertical-align:middle; margin-left:3px; }
    .complexity-fill { height:100%; border-radius:4px; background: var(--blue-mid); }
    .ctx-badge { font-size:.62rem; background:var(--blue-light); color:var(--blue); padding:2px 6px; border-radius:20px; font-weight:600; }
    /* Recommendation carousel */
    .rec-carousel { margin-top: .25rem; width: 100%; max-width: 100%; }
    .rec-viewport { width: 100%; max-width: 100%; overflow: hidden; border-radius: var(--radius); }
    .rec-track { display: flex; width: 100%; transition: transform 0.4s cubic-bezier(.4,0,.2,1); will-change: transform; }
    .rec-slide { flex: 0 0 100%; width: 100%; max-width: 100%; min-width: 0; box-sizing: border-box; }
    .rec-slide .article-card { width: 100%; max-width: 100%; min-height: 220px; padding: 1.1rem 1.2rem; border-left-width: 5px; box-sizing: border-box; }
    .carousel-nav { display: flex; align-items: center; justify-content: center; gap: .75rem; margin-top: .85rem; }
    .carousel-btn { width: 36px; height: 36px; border-radius: 50%; border: 1.5px solid var(--gray-200); background: white; color: var(--blue); font-size: 1.25rem; line-height: 1; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all .15s; flex-shrink: 0; box-shadow: var(--shadow); }
    .carousel-btn:hover:not(:disabled) { background: var(--blue); color: white; border-color: var(--blue); }
    .carousel-btn:disabled { opacity: .3; cursor: not-allowed; box-shadow: none; }
    .carousel-footer { display: flex; flex-direction: column; align-items: center; gap: .45rem; margin-top: .5rem; }
    .carousel-dots { display: flex; gap: .45rem; justify-content: center; }
    .carousel-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--gray-200); border: none; cursor: pointer; padding: 0; transition: all .25s; }
    .carousel-dot.active { background: var(--green); width: 24px; border-radius: 4px; }
    .carousel-counter { font-size: .72rem; color: var(--gray-400); font-weight: 500; letter-spacing: .3px; }
    /* Modal */
    .modal-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,.45); z-index:200; align-items:center; justify-content:center; padding:1rem; }
    .modal-overlay.open { display:flex; }
    .modal { background:white; border-radius: 14px; box-shadow:0 20px 60px rgba(0,0,0,.2); max-width:540px; width:100%; max-height:90vh; overflow-y:auto; }
    .modal-header { padding:1.2rem 1.4rem .8rem; border-bottom:1px solid var(--gray-100); display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; }
    .modal-header h2 { font-size:1rem; font-weight:700; line-height:1.4; color:var(--gray-800); }
    .modal-close { background:none; border:none; font-size:1.3rem; cursor:pointer; color:var(--gray-400); flex-shrink:0; padding:0; line-height:1; }
    .modal-close:hover { color:var(--gray-800); }
    .modal-body { padding:1rem 1.4rem 1.4rem; display:flex; flex-direction:column; gap:.9rem; }
    .modal-meta { display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; }
    .modal-summary { font-size:.85rem; color:var(--gray-600); line-height:1.65; }
    .modal-stats { display:flex; gap:1.2rem; font-size:.78rem; color:var(--gray-600); background:var(--gray-50); border-radius:8px; padding:.6rem .9rem; }
    .modal-stat { display:flex; flex-direction:column; gap:.15rem; }
    .modal-stat label { font-size:.65rem; text-transform:uppercase; letter-spacing:.6px; color:var(--gray-400); font-weight:600; }
    .modal-stat span { font-weight:600; color:var(--gray-800); }
    .modal-similar-btn { align-self:flex-start; }
    .modal-similar-list { display:flex; flex-direction:column; gap:.4rem; margin-top:.2rem; }
    .modal-sim-item { font-size:.78rem; padding:.45rem .65rem; background:var(--gray-50); border-radius:6px; cursor:pointer; border-left:3px solid var(--blue-light); color:var(--gray-700); transition:background .15s; }
    .modal-sim-item:hover { background:var(--blue-light); }
    .site-footer { margin-top: 2rem; background: var(--blue); color: rgba(255,255,255,.85); text-align: center; padding: .65rem 1rem; font-size: .72rem; border-top: 4px solid var(--green); }
    .site-footer strong { color: var(--green); font-weight: 600; }
  </style>
</head>
<body>
<header>
  <div class="logo">Med<span>X</span></div>
  <span class="header-badge">Inspired by coliquio</span>
</header>
<div class="context-toast" id="contextToast" role="status" aria-live="polite">
  <div class="context-toast-inner">
    <span class="context-toast-icon" id="contextToastIcon"></span>
    <div class="context-toast-body">
      <div class="context-toast-title" id="contextToastTitle"></div>
      <div class="context-toast-msg" id="contextToastMsg"></div>
    </div>
    <button type="button" class="context-toast-close" onclick="dismissContextToast()" aria-label="Dismiss">×</button>
  </div>
  <div class="context-toast-progress" id="contextToastProgress"><span></span></div>
</div>
<div class="layout">
  <aside class="sidebar">
    <div class="card">
      <div class="card-header">Select Doctor</div>
      <div class="card-body" style="display:flex;flex-direction:column;gap:.75rem;">
        <select id="doctorSelect" onchange="onDoctorChange()">
          <option value="">— Choose a doctor —</option>
        </select>
        <button class="btn btn-accent" id="getRecsBtn" disabled onclick="fetchRecommendations()">Get Recommendations</button>
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
      <div class="card-header">About MedX</div>
      <div class="card-body" style="font-size:.78rem;color:var(--gray-600);line-height:1.6;">
        <p>The right medical article, for the right doctor, at the right time.</p>
        <p style="margin-top:.65rem;">UI inspired from <strong style="color:var(--green-dark);">coliquio</strong> — the German HCP community platform.</p>
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
          <div class="rec-carousel" id="recCarousel">
            <div class="rec-viewport">
              <div class="rec-track" id="recTrack"></div>
            </div>
            <div class="carousel-nav">
              <button type="button" class="carousel-btn" id="recPrev" onclick="recPrev()" aria-label="Previous">‹</button>
              <div class="carousel-dots" id="recDots"></div>
              <button type="button" class="carousel-btn" id="recNext" onclick="recNext()" aria-label="Next">›</button>
            </div>
            <div class="carousel-footer">
              <div class="carousel-counter" id="recCounter"></div>
            </div>
          </div>
        </div>
        <div id="allPanel" style="display:none;">
          <div class="article-grid" id="allGrid"></div>
        </div>
        <div id="historyPanel" style="display:none;">
          <div class="read-list" id="readList"></div>
        </div>
      </div>
    </div>
  </main>
</div>
<footer class="site-footer">MedX · Hybrid recommender PoC · UI inspired from <strong>coliquio</strong></footer>
<!-- Article Modal -->
<div class="modal-overlay" id="modalOverlay" onclick="closeModal(event)">
  <div class="modal" id="modal">
    <div class="modal-header">
      <h2 id="modalTitle"></h2>
      <button class="modal-close" onclick="closeModal()">✕</button>
    </div>
    <div class="modal-body">
      <div class="modal-meta" id="modalMeta"></div>
      <div class="modal-stats" id="modalStats"></div>
      <p class="modal-summary" id="modalSummary"></p>
      <div class="tags" id="modalTags"></div>
      <button class="btn modal-similar-btn" onclick="loadModalSimilar()" id="similarBtn" style="background:var(--gray-100);color:var(--gray-700);font-size:.78rem;padding:.4rem .85rem;">
        Show similar articles
      </button>
      <div class="modal-similar-list" id="modalSimilarList" style="display:none;"></div>
    </div>
  </div>
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
  const hour = new Date().getHours();
  setLoading(true);
  switchTab('rec');
  const res = await fetch(`${API}/api/recommend/${id}?n=5&alpha=${alpha}&hour=${hour}`);
  const data = await res.json();
  setLoading(false);
  document.getElementById('emptyState').style.display = 'none';
  document.getElementById('recPanel').style.display = 'block';
  document.getElementById('recSubtitle').textContent =
    `Top recommendations for ${data.doctor.name}`;

  // Show context toast
  if (data.context) showContextToast(data.context);

  renderRecCarousel(data.recommendations);
}

let contextToastTimer = null;
const CONTEXT_TOAST_MS = 5000;

function dismissContextToast() {
  const toast = document.getElementById('contextToast');
  if (!toast) return;
  toast.classList.remove('show');
  if (contextToastTimer) {
    clearTimeout(contextToastTimer);
    contextToastTimer = null;
  }
}

function showContextToast(c) {
  dismissContextToast();
  const toast = document.getElementById('contextToast');
  const complexityPct = Math.round((c.ideal_complexity ?? 0.5) * 100);
  document.getElementById('contextToastIcon').textContent = c.icon || '🕐';
  document.getElementById('contextToastTitle').textContent = c.label || 'Time context';
  document.getElementById('contextToastMsg').innerHTML =
    `Showing articles that fit your reading time right now — <strong>≤${c.max_reading_min} min</strong> · complexity match (~${complexityPct}%)`;
  const bar = document.querySelector('#contextToastProgress span');
  if (bar) {
    bar.style.animation = 'none';
    void bar.offsetWidth;
    bar.style.animation = `toast-progress ${CONTEXT_TOAST_MS}ms linear forwards`;
  }
  requestAnimationFrame(() => toast.classList.add('show'));
  contextToastTimer = setTimeout(dismissContextToast, CONTEXT_TOAST_MS);
}

let recSlides = [];
let recIndex = 0;

function renderRecCarousel(recommendations) {
  recSlides = recommendations.slice(0, 5);
  recIndex = 0;
  const track = document.getElementById('recTrack');
  const dots = document.getElementById('recDots');
  track.innerHTML = '';
  dots.innerHTML = '';
  recSlides.forEach((art, i) => {
    const slide = document.createElement('div');
    slide.className = 'rec-slide';
    slide.appendChild(buildArticleCard(art, true, i === 0));
    track.appendChild(slide);
    const dot = document.createElement('button');
    dot.type = 'button';
    dot.className = 'carousel-dot' + (i === 0 ? ' active' : '');
    dot.setAttribute('aria-label', `Slide ${i + 1}`);
    dot.onclick = () => goToRecSlide(i);
    dots.appendChild(dot);
  });
  updateRecCarousel();
}

function goToRecSlide(i) {
  if (!recSlides.length) return;
  recIndex = Math.max(0, Math.min(i, recSlides.length - 1));
  updateRecCarousel();
}

function recPrev() { goToRecSlide(recIndex - 1); }
function recNext() { goToRecSlide(recIndex + 1); }

function syncRecSlideWidths() {
  const viewport = document.querySelector('.rec-viewport');
  if (!viewport) return 0;
  const w = viewport.clientWidth;
  document.querySelectorAll('.rec-slide').forEach(slide => {
    slide.style.flexBasis = `${w}px`;
    slide.style.width = `${w}px`;
  });
  return w;
}

function updateRecCarousel() {
  const track = document.getElementById('recTrack');
  const slideWidth = syncRecSlideWidths();
  track.style.transform = slideWidth
    ? `translateX(-${recIndex * slideWidth}px)`
    : `translateX(-${recIndex * 100}%)`;
  document.querySelectorAll('.carousel-dot').forEach((d, i) => {
    d.classList.toggle('active', i === recIndex);
  });
  const counter = document.getElementById('recCounter');
  counter.textContent = recSlides.length ? `${recIndex + 1} of ${recSlides.length}` : '';
  document.getElementById('recPrev').disabled = recIndex === 0;
  document.getElementById('recNext').disabled = recIndex >= recSlides.length - 1;
}

window.addEventListener('resize', () => {
  if (recSlides.length) updateRecCarousel();
});

function switchTab(tab) {
  currentTab = tab;
  ['rec','all','history'].forEach(t => {
    document.getElementById(`tab${t.charAt(0).toUpperCase()+t.slice(1)}`).classList.remove('active');
    const panel = document.getElementById(`${t}Panel`);
    if (panel) panel.style.display = 'none';
  });
  document.getElementById(`tab${tab.charAt(0).toUpperCase()+tab.slice(1)}`).classList.add('active');
  if (tab === 'rec') {
    const hasRecs = document.getElementById('recTrack').children.length > 0;
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

function buildArticleCard(art, showScore = false, isTopRec = false) {
  const card = document.createElement('div');
  card.className = 'article-card';
  card.dataset.id = art.id;
  const scoreVal = art.score ?? art.similarity ?? 0;
  const scorePct = Math.round(scoreVal * 100);
  const complexPct = art.complexity_score != null ? Math.round(art.complexity_score * 100) : null;
  const readTime = art.reading_time_minutes ?? null;
  const ctxBadge = art.context_icon ? `<span class="ctx-badge">${art.context_icon} ${art.context_label}</span>` : '';
  const typeLabel = art.type.replace(/_/g, ' ');
  const typeHighlight = ['education', 'guidelines'].includes(art.type);
  const typeHtml = typeHighlight
    ? `<span class="type-pill">${typeLabel}</span>`
    : typeLabel;
  card.innerHTML = `
    ${isTopRec ? '<div class="rec-rank">Top pick</div>' : ''}
    <div class="art-type">${typeHtml}${ctxBadge ? ' · ' + ctxBadge : ''}</div>
    <div class="art-title">${art.title}</div>
    <div class="art-summary">${art.summary}</div>
    <div class="tags">
      <span class="tag specialty-tag">${art.specialty.replace(/_/g,' ')}</span>
      ${art.tags.slice(0,3).map(t => `<span class="tag">${t}</span>`).join('')}
    </div>
    ${readTime != null ? `
    <div class="art-meta">
      <span>⏱ ${readTime} min read</span>
      ${complexPct != null ? `<span>Complexity <div class="complexity-bar" style="display:inline-block"><div class="complexity-fill" style="width:${complexPct}%"></div></div></span>` : ''}
    </div>` : ''}
    ${showScore ? `<div class="art-score"><span>Relevance</span><div class="score-bar"><div class="score-fill" style="width:${scorePct}%"></div></div><span>${scorePct}%</span></div>` : ''}
  `;
  card.onclick = () => openModal(art);
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

let _modalArticle = null;

function openModal(art) {
  _modalArticle = art;
  document.getElementById('modalTitle').textContent = art.title;

  // Meta badges
  const meta = document.getElementById('modalMeta');
  meta.innerHTML = `
    <span class="tag specialty-tag">${art.specialty.replace(/_/g,' ')}</span>
    <span class="tag">${art.type.replace(/_/g,' ')}</span>
    ${art.context_icon ? `<span class="ctx-badge">${art.context_icon} ${art.context_label}</span>` : ''}
  `;

  // Stats
  const complexPct = art.complexity_score != null ? Math.round(art.complexity_score * 100) : '—';
  document.getElementById('modalStats').innerHTML = `
    <div class="modal-stat"><label>Read time</label><span>⏱ ${art.reading_time_minutes ?? '—'} min</span></div>
    <div class="modal-stat"><label>Complexity</label><span>${complexPct}%</span></div>
    ${art.score != null ? `<div class="modal-stat"><label>Relevance</label><span>${Math.round(art.score*100)}%</span></div>` : ''}
  `;

  document.getElementById('modalSummary').textContent = art.summary;

  // Tags
  const tagsEl = document.getElementById('modalTags');
  tagsEl.innerHTML = art.tags.map(t => `<span class="tag">${t}</span>`).join('');

  // Reset similar section
  document.getElementById('modalSimilarList').style.display = 'none';
  document.getElementById('modalSimilarList').innerHTML = '';
  document.getElementById('similarBtn').textContent = 'Show similar articles';

  document.getElementById('modalOverlay').classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closeModal(e) {
  if (e && e.target !== document.getElementById('modalOverlay') && e.type === 'click') return;
  document.getElementById('modalOverlay').classList.remove('open');
  document.body.style.overflow = '';
}

async function loadModalSimilar() {
  if (!_modalArticle) return;
  const btn = document.getElementById('similarBtn');
  btn.textContent = 'Loading…';
  const res = await fetch(`${API}/api/articles/${_modalArticle.id}/similar?n=4`);
  const data = await res.json();
  const list = document.getElementById('modalSimilarList');
  list.innerHTML = '<div style="font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--gray-400);margin-bottom:.3rem;">Similar articles</div>';
  data.similar.forEach(art => {
    const item = document.createElement('div');
    item.className = 'modal-sim-item';
    item.innerHTML = `<strong>${art.title}</strong><br><span style="color:var(--gray-400)">${art.specialty.replace(/_/g,' ')} · ⏱ ${art.reading_time_minutes} min · ${Math.round(art.similarity*100)}% similar</span>`;
    item.onclick = () => openModal(art);
    list.appendChild(item);
  });
  list.style.display = 'flex';
  btn.style.display = 'none';
}

function setLoading(on) {
  document.getElementById('spinner').style.display = on ? 'block' : 'none';
  document.getElementById('recPanel').style.display = 'none';
  document.getElementById('emptyState').style.display = 'none';
  if (on) dismissContextToast();
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
    n: int = Query(5, ge=1, le=5, description="Number of recommendations (max 5)"),
    alpha: float = 0.5,
    exclude_read: bool = True,
    hour: int | None = None,
    use_ranker: bool = Query(True, description="Use LightGBM ranker when model is loaded"),
):
    """
    Get personalised article recommendations for a doctor.

    - **alpha**: content-based weight 0–1 (0 = pure collaborative, 1 = pure content)
    - **hour**: 0–23 hour of day for time-context re-ranking (omit to skip context)
    """
    doc = get_rec().get_doctor(doctor_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Doctor not found")

    effective_hour = hour if hour is not None else datetime.now(timezone.utc).hour
    slot = get_time_slot(effective_hour)
    recs, ranker_mode = get_rec().recommend(
        doctor_id,
        n=n,
        alpha=alpha,
        exclude_read=exclude_read,
        hour=effective_hour,
        use_ranker=use_ranker,
    )
    return {
        "doctor": doc,
        "recommendations": recs,
        "ranker": ranker_mode,
        "context": {
            "hour":  effective_hour,
            "label": slot["label"],
            "icon":  slot["icon"],
            "ideal_complexity": slot["ideal_complexity"],
            "max_reading_min":  slot["max_reading_min"],
        },
    }


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
    rec = get_rec()
    if rec.ranker.is_loaded:
        ranker_backend = "lightgbm"
    elif rec.sk_ranker.is_loaded:
        ranker_backend = "sklearn"
    else:
        ranker_backend = "hybrid"
    ranker_detail = {}
    if not rec.ranker.is_loaded:
        ranker_detail["lightgbm"] = "not installed on Vercel"
    if not rec.sk_ranker.is_loaded and rec.sk_ranker.load_error:
        ranker_detail["sklearn_error"] = rec.sk_ranker.load_error
    return {
        "status": "ok",
        "model": "hybrid (TF-IDF + numpy SVD) + learned ranker",
        "ranker_backend": ranker_backend,
        "ranker_loaded": rec.ranker_loaded,
        "ranker_detail": ranker_detail or None,
        "event_logs": len(rec.event_logs_df),
        "version": "0.2.1",
    }
