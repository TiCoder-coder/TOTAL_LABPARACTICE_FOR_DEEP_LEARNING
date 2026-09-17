<!-- Refactored visual Markdown. CSS/SVG only; no JavaScript. Scientific content preserved from source. -->
<style>
:root{
  --pf-ink:#101828; --pf-muted:#475467; --pf-blue:#0b63ce; --pf-blue-2:#2f80ed;
  --pf-cyan:#4cc9f0; --pf-yellow:#f5c84b; --pf-yellow-soft:#fff6c8;
  --pf-paper:#ffffff; --pf-soft:#f6faff; --pf-line:#d8e7f5; --pf-green:#10b981;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
.pf-shell{max-width:1200px;margin:0 auto;padding:24px 20px 0;background:linear-gradient(180deg,#fff 0%,#f7fbff 45%,#fffdf4 100%);color:var(--pf-ink);font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;line-height:1.72}
.pf-hero{position:relative;isolation:isolate;overflow:hidden;border:1px solid #cfe2f7;border-radius:30px;padding:38px 38px 30px;background:radial-gradient(circle at 84% 12%,rgba(245,200,75,.32),transparent 23%),radial-gradient(circle at 10% 8%,rgba(76,201,240,.23),transparent 28%),linear-gradient(135deg,#fff 0%,#f3f9ff 58%,#fff9dc 100%);box-shadow:0 22px 65px rgba(15,81,145,.12)}
.pf-hero:before,.pf-hero:after{content:"";position:absolute;border-radius:50%;z-index:-1}.pf-hero:before{width:230px;height:230px;right:-78px;bottom:-102px;border:1px solid rgba(11,99,206,.18);animation:pfPulse 5s ease-in-out infinite}.pf-hero:after{width:115px;height:115px;right:-14px;bottom:-36px;border:1px solid rgba(245,200,75,.58);animation:pfPulse 4s ease-in-out infinite reverse}
.pf-kicker{font-size:12px;font-weight:850;letter-spacing:.18em;text-transform:uppercase;color:#0758b3}.pf-title{margin:7px 0 10px!important;color:#0b1220!important;font-size:clamp(32px,5vw,54px)!important;line-height:1.05!important;border:0!important}.pf-subtitle{max-width:880px;margin:0;color:#344054;font-size:16px}.pf-subtitle strong{color:#0b63ce}
.pf-badges{display:flex;flex-wrap:wrap;gap:10px;margin-top:22px}.pf-badge{display:inline-flex;align-items:center;gap:8px;padding:8px 12px;border:1px solid #d8e7f6;border-radius:999px;background:rgba(255,255,255,.9);font-size:13px;font-weight:750;box-shadow:0 6px 18px rgba(15,81,145,.08);transition:transform .22s ease,box-shadow .22s ease}.pf-badge:hover{transform:translateY(-3px);box-shadow:0 10px 24px rgba(15,81,145,.14)}.pf-badge svg{width:19px;height:19px;display:block}
.pf-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:22px}.pf-stat{padding:14px 15px;border-radius:16px;background:rgba(255,255,255,.82);border:1px solid #dce9f6;backdrop-filter:blur(4px)}.pf-stat-value{font-size:23px;font-weight:900;line-height:1;color:#0b63ce}.pf-stat:nth-child(4) .pf-stat-value{color:#a36b00}.pf-stat-label{margin-top:6px;font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#667085}
.pf-flow{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;gap:10px;align-items:center;margin:22px 0 6px}.pf-flow-node{padding:13px 14px;border-radius:14px;border:1px solid #d8e6f4;background:#fff;font-weight:800;text-align:center;box-shadow:0 5px 16px rgba(15,81,145,.06)}.pf-flow-node small{display:block;margin-top:2px;color:#667085;font-weight:650}.pf-flow-arrow{font-size:22px;color:#0b63ce;animation:pfArrow 1.8s ease-in-out infinite}
.pf-document{padding:10px 0 40px}.pf-section{position:relative;margin:20px 0;padding:25px 27px;background:rgba(255,255,255,.96);border:1px solid var(--pf-line);border-radius:21px;box-shadow:0 8px 28px rgba(15,81,145,.065);transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease}.pf-section:hover{transform:translateY(-2px);box-shadow:0 15px 36px rgba(15,81,145,.105);border-color:#bdd9f3}
.pf-section.pf-phase-card:before{content:"";position:absolute;left:0;top:20px;bottom:20px;width:4px;border-radius:0 6px 6px 0;background:linear-gradient(180deg,var(--pf-blue),var(--pf-cyan))}.pf-section.pf-group-modeling:before{background:linear-gradient(180deg,#367de7,#78a8ff)}.pf-section.pf-group-sweeps:before{background:linear-gradient(180deg,#f5c84b,#f59e0b)}
.pf-heading{scroll-margin-top:20px;color:#101828}.pf-h1{display:none}.pf-h2{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:0 0 18px!important;padding:0 0 12px;border-bottom:3px solid transparent;border-image:linear-gradient(90deg,var(--pf-blue),var(--pf-cyan),var(--pf-yellow),transparent) 1;font-size:27px!important;line-height:1.25!important}.pf-h3{margin:25px 0 10px!important;color:#0b5eb8!important;font-size:20px!important}.pf-h3:before{content:"◆";margin-right:8px;color:var(--pf-yellow);font-size:.72em}.pf-phase-pill{display:inline-flex;align-items:center;padding:5px 8px;border-radius:8px;background:#eaf4ff;color:#0758b3;font-size:10px;font-weight:900;letter-spacing:.1em;white-space:nowrap}.pf-group-sweeps .pf-phase-pill{background:#fff5c9;color:#855b00}
.pf-toc-card{background:linear-gradient(135deg,#fbfdff,#f3f9ff 68%,#fffaf0)}.pf-toc-card:after{content:"Navigation Map";position:absolute;top:18px;right:22px;padding:5px 9px;border-radius:999px;background:#fff3b8;color:#7c5700;font-size:10px;font-weight:900;letter-spacing:.08em;text-transform:uppercase}.pf-toc-list>li{margin:10px 0}.pf-toc-list ul{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px 18px;margin-top:8px}.pf-toc-list a{font-weight:650!important}
.pf-summary-card{background:radial-gradient(circle at 95% 10%,rgba(245,200,75,.20),transparent 28%),linear-gradient(135deg,#fff,#f4f9ff)}
.pf-note{margin:18px 0!important;padding:16px 18px!important;border-left:4px solid var(--pf-blue)!important;background:linear-gradient(90deg,#edf6ff,#fffdf2)!important;border-radius:0 13px 13px 0;color:#26364a}.pf-rule{border:0!important;height:1px!important;background:linear-gradient(90deg,transparent,#93c5fd,#facc15,transparent)!important;margin:28px 0!important}.pf-link{color:#075fbe!important;text-decoration:none!important;font-weight:650}.pf-link:hover{text-decoration:underline!important;text-decoration-thickness:1.5px!important}.pf-inline-code{padding:.14em .42em;border:1px solid #d9e7f4;border-radius:6px;background:#f3f8fc;color:#854d0e;font-size:.92em}.pf-code{overflow:auto;padding:16px 18px;border:1px solid #cfe0ef;border-radius:14px;background:#f7fbff;box-shadow:inset 4px 0 0 #4cc9f0}.pf-code code{background:transparent!important;color:#111827!important}
.pf-table-wrap{overflow-x:auto;margin:18px 0;border:1px solid #d7e6f4;border-radius:14px;box-shadow:0 5px 18px rgba(15,81,145,.06)}.pf-table{width:100%;border-collapse:collapse;background:#fff;font-size:14px}.pf-table th{padding:11px 12px;background:linear-gradient(90deg,#eaf5ff,#fff6c9);color:#172033;text-align:left;border-bottom:1px solid #cbddeb}.pf-table td{padding:10px 12px;border-bottom:1px solid #edf2f7;vertical-align:top}.pf-table tr:last-child td{border-bottom:0}.pf-table tbody tr:hover{background:#f8fcff}
ul,ol{padding-left:24px}li::marker{color:#0b63ce}strong{color:#111827}
.pf-handoff-link{display:block;color:inherit!important;text-decoration:none!important}.pf-handoff{position:relative;overflow:hidden;margin:24px 0 0;padding:23px 24px;border-radius:20px;border:1px solid #cfe2f7;background:linear-gradient(135deg,#eef7ff 0%,#fff 58%,#fff5c9 100%);box-shadow:0 10px 28px rgba(15,81,145,.08);transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease}.pf-handoff-link:hover .pf-handoff{transform:translateY(-3px);border-color:#8fc5f3;box-shadow:0 16px 38px rgba(15,81,145,.14)}.pf-handoff-grid{display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:16px}.pf-handoff-icon{width:50px;height:50px;border-radius:15px;display:grid;place-items:center;background:#fff;border:1px solid #d7e6f4;box-shadow:0 6px 16px rgba(15,81,145,.08)}.pf-handoff h3{margin:0!important;color:#101828!important}.pf-handoff p{margin:3px 0 0;color:#475467}.pf-handoff-arrow{font-size:28px;color:#0b63ce;animation:pfArrow 1.8s ease-in-out infinite}
.pf-footer{position:relative;overflow:hidden;margin:24px -20px 0;padding:66px 24px 92px;text-align:center;background:linear-gradient(180deg,#f8fcff,#e9f6ff 54%,#fff4bf);border-top:1px solid #cfe4f7}.pf-footer h3{margin:0;color:#101828;font-size:24px}.pf-footer p{margin:6px 0;color:#475467}.pf-floaters{display:flex;justify-content:center;gap:17px;margin:22px 0 0}.pf-floater{width:38px;height:38px;display:grid;place-items:center;border-radius:12px;background:#fff;border:1px solid #d9e7f4;box-shadow:0 7px 18px rgba(15,81,145,.10);animation:pfFloat 3.2s ease-in-out infinite}.pf-floater:nth-child(2){animation-delay:.35s}.pf-floater:nth-child(3){animation-delay:.7s}.pf-floater:nth-child(4){animation-delay:1.05s}.pf-floater:nth-child(5){animation-delay:1.4s}.pf-floater svg{width:22px;height:22px}.pf-wave{position:absolute;left:-1%;right:-1%;bottom:-2px;width:102%;height:82px;opacity:.54}.pf-wave path:first-child{animation:pfWaveA 5.5s ease-in-out infinite alternate}.pf-wave path:last-child{animation:pfWaveB 6.5s ease-in-out infinite alternate}
@keyframes pfPulse{0%,100%{transform:scale(1);opacity:.65}50%{transform:scale(1.11);opacity:1}}@keyframes pfFloat{0%,100%{transform:translateY(0) rotate(0)}50%{transform:translateY(-11px) rotate(4deg)}}@keyframes pfArrow{0%,100%{transform:translateX(0);opacity:.65}50%{transform:translateX(5px);opacity:1}}@keyframes pfWaveA{from{transform:translateX(-9px)}to{transform:translateX(9px)}}@keyframes pfWaveB{from{transform:translateX(8px)}to{transform:translateX(-8px)}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important;scroll-behavior:auto!important}}
@media(max-width:760px){.pf-shell{padding:12px 10px 0}.pf-hero{padding:26px 20px;border-radius:21px}.pf-stats{grid-template-columns:repeat(2,minmax(0,1fr))}.pf-flow{grid-template-columns:1fr}.pf-flow-arrow{transform:rotate(90deg)!important;text-align:center}.pf-section{padding:19px 16px;border-radius:16px}.pf-h2{font-size:23px!important}.pf-toc-list ul{grid-template-columns:1fr}.pf-footer{margin-left:-10px;margin-right:-10px}.pf-handoff-grid{grid-template-columns:auto 1fr}.pf-handoff-arrow{display:none}}
</style>
<div class="pf-shell">
<div class="pf-hero">
<div class="pf-kicker">CourseWork · Verified V1 Flow</div>
<h1 class="pf-title">Phase 1 → 33 · Current Flow Detail</h1>
<p class="pf-subtitle">Bản trực quan hóa cho toàn bộ flow từ <strong>Environment & Data Pipeline</strong> → <strong>Model Foundations</strong> → <strong>Sweeps S1–S11</strong>.</p>
<div class="pf-badges">
<span class="pf-badge"><svg viewBox="0 0 24 24" aria-hidden="true"><path fill="#3776AB" d="M12 2c-4 0-4 2-4 4v2h8v1H6c-2 0-4 1-4 4s2 4 4 4h2v-3c0-2 2-4 4-4h6c2 0 4-2 4-4s-2-4-4-4z"/><circle cx="10" cy="5" r="1" fill="#fff"/><path fill="#FFD43B" d="M12 22c4 0 4-2 4-4v-2H8v-1h10c2 0 4-1 4-4s-2-4-4-4h-2v3c0 2-2 4-4 4H6c-2 0-4 2-4 4s2 4 4 4z"/><circle cx="14" cy="19" r="1" fill="#fff"/></svg>Python</span>
<span class="pf-badge"><svg viewBox="0 0 24 24"><path d="M12 2 4 6v8l8 4 8-4V6z" fill="#ee4c2c"/><path d="M8 9h8v6H8z" fill="#fff" opacity=".92"/></svg>PyTorch</span>
<span class="pf-badge"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="6" fill="none" stroke="#f59e0b" stroke-width="2"/><circle cx="12" cy="3" r="1.6" fill="#6b7280"/><circle cx="12" cy="21" r="1.6" fill="#6b7280"/></svg>Jupyter</span>
<span class="pf-badge"><svg viewBox="0 0 24 24"><path d="M13 2 5 13h6l-1 9 9-13h-6z" fill="#f5c84b" stroke="#9a6700" stroke-width=".7"/></svg>Transformer</span>
<span class="pf-badge"><svg viewBox="0 0 24 24"><path d="M3 17c4-8 6 2 9-6s5 2 9-5" fill="none" stroke="#0b63ce" stroke-width="2" stroke-linecap="round"/><circle cx="3" cy="17" r="1.4" fill="#4cc9f0"/><circle cx="12" cy="11" r="1.4" fill="#4cc9f0"/><circle cx="21" cy="6" r="1.4" fill="#f5c84b"/></svg>Time Series</span>
<span class="pf-badge"><svg viewBox="0 0 24 24"><path d="M4 6h16v12H4z" fill="none" stroke="#0b63ce" stroke-width="1.8"/><path d="M4 10h16M9 6v12M15 6v12" stroke="#0b63ce" stroke-width="1.2"/></svg>Data Pipeline</span>
<span class="pf-badge"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" fill="#e8fff7" stroke="#10b981" stroke-width="1.5"/><path d="m8 12 2.5 2.5L16.5 8.5" fill="none" stroke="#10b981" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>Verified Artifacts</span>
</div>
<div class="pf-stats">
  <div class="pf-stat"><div class="pf-stat-value">33</div><div class="pf-stat-label">Phases</div></div>
  <div class="pf-stat"><div class="pf-stat-value">153</div><div class="pf-stat-label">Notebook Cells</div></div>
  <div class="pf-stat"><div class="pf-stat-value">3</div><div class="pf-stat-label">Flow Groups</div></div>
  <div class="pf-stat"><div class="pf-stat-value">V1</div><div class="pf-stat-label">Verified Lineage</div></div>
</div>
<div class="pf-flow" aria-label="Flow overview">
  <div class="pf-flow-node">Foundation<small>Phase 1–10</small></div><div class="pf-flow-arrow">→</div>
  <div class="pf-flow-node">Modeling<small>Phase 11–22</small></div><div class="pf-flow-arrow">→</div>
  <div class="pf-flow-node">Sweeps S1–S11<small>Phase 23–33</small></div>
</div>
</div>
<div class="pf-document"><h1 class="pf-heading pf-h1" id="phase-1-to-33--current-flow-detail">PHASE 1 TO 33 — Current Flow Detail</h1>
<blockquote class="pf-note">
<p><a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">CourseWork.ipynb</a> hiện có 153 cells. Mapping đầy đủ nằm tại <a class="pf-link" href="../link&amp;discussion_to_result/NOTEBOOK_CELLS_WALKTHROUGH.md">NOTEBOOK_CELLS_WALKTHROUGH.md</a>.</p>
</blockquote>
<section class="pf-section pf-toc-card"><h2 class="pf-heading pf-h2 pf-toc-title" id="mục-lục">Mục lục</h2>
<ul class="pf-toc-list">
<li><strong>Foundation và data pipeline</strong><ul>
<li><a class="pf-link" href="#phase-1---environment">Phase 1 — Environment</a></li>
<li><a class="pf-link" href="#phase-2---data-acquisition">Phase 2 — Data Acquisition</a></li>
<li><a class="pf-link" href="#phase-3---schema-audit">Phase 3 — Schema Audit</a></li>
<li><a class="pf-link" href="#phase-4---temporal-integrity-audit">Phase 4 — Temporal Integrity Audit</a></li>
<li><a class="pf-link" href="#phase-5---chronological-split">Phase 5 — Chronological Split</a></li>
<li><a class="pf-link" href="#phase-6---exploratory-data-analysis">Phase 6 — Exploratory Data Analysis</a></li>
<li><a class="pf-link" href="#phase-7---feature-engineering">Phase 7 — Feature Engineering</a></li>
<li><a class="pf-link" href="#phase-8---feature-set-variants">Phase 8 — Feature-Set Variants</a></li>
<li><a class="pf-link" href="#phase-9---train-only-scaling">Phase 9 — Train-Only Scaling</a></li>
<li><a class="pf-link" href="#phase-10---window-builder">Phase 10 — Window Builder</a></li>
<li><a class="pf-link" href="#phase-11---dataloaders">Phase 11 — DataLoaders</a></li>
</ul>
</li>
<li><strong>Metrics, registry và model foundations</strong><ul>
<li><a class="pf-link" href="#phase-12---shared-metrics">Phase 12 — Shared Metrics</a></li>
<li><a class="pf-link" href="#phase-13---experiment-registry">Phase 13 — Experiment Registry</a></li>
<li><a class="pf-link" href="#phase-14---persistence-baseline">Phase 14 — Persistence Baseline</a></li>
<li><a class="pf-link" href="#phase-15---lstm-implementation">Phase 15 — LSTM Implementation</a></li>
<li><a class="pf-link" href="#phase-16---transformer-implementation">Phase 16 — Transformer Implementation</a></li>
<li><a class="pf-link" href="#phase-17---attention-aware-encoder-verification">Phase 17 — Attention-Aware Encoder Verification</a></li>
<li><a class="pf-link" href="#phase-18---forward-pass-sanity-tests">Phase 18 — Forward-Pass Sanity Tests</a></li>
<li><a class="pf-link" href="#phase-19---baseline-training-engine">Phase 19 — Baseline Training Engine</a></li>
<li><a class="pf-link" href="#phase-20---lstm-baseline-run">Phase 20 — LSTM Baseline Run</a></li>
<li><a class="pf-link" href="#phase-21---transformer-b0-run">Phase 21 — Transformer B0 Run</a></li>
<li><a class="pf-link" href="#phase-22---learning-curve-diagnostics">Phase 22 — Learning-Curve Diagnostics</a></li>
</ul>
</li>
<li><strong>Sweeps S1–S11</strong><ul>
<li><a class="pf-link" href="#phase-23---s1-feature-set-sweep">Phase 23 — S1 Feature-Set Sweep</a></li>
<li><a class="pf-link" href="#phase-24---s2-time-feature-sweep">Phase 24 — S2 Time-Feature Sweep</a></li>
<li><a class="pf-link" href="#phase-25---s3-target-scaling-sweep">Phase 25 — S3 Target-Scaling Sweep</a></li>
<li><a class="pf-link" href="#phase-26---s4-lookback-sweep">Phase 26 — S4 Lookback Sweep</a></li>
<li><a class="pf-link" href="#phase-27---s5-pooling-sweep">Phase 27 — S5 Pooling Sweep</a></li>
<li><a class="pf-link" href="#phase-28---s6-activation-sweep">Phase 28 — S6 Activation Sweep</a></li>
<li><a class="pf-link" href="#phase-29---s7-batch-size-sweep">Phase 29 — S7 Batch-Size Sweep</a></li>
<li><a class="pf-link" href="#phase-30---s8-learning-rate-sweep">Phase 30 — S8 Learning-Rate Sweep</a></li>
<li><a class="pf-link" href="#phase-31---s9-weight-decay-sweep">Phase 31 — S9 Weight-Decay Sweep</a></li>
<li><a class="pf-link" href="#phase-32---s10-dropout-sweep">Phase 32 — S10 Dropout Sweep</a></li>
<li><a class="pf-link" href="#transformer-configuration-after-phase-33">Transformer Configuration after Phase 33</a></li>
</ul>
</li>
<li><a class="pf-link" href="#tổng-kết-phases-133">Tổng kết Phases 1–33</a></li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-foundation" data-phase="1"><h2 class="pf-heading pf-h2" id="phase-1---environment"><span aria-hidden="true" class="pf-phase-pill">PHASE 1</span>Phase 1 - Environment</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 4–6</a> — heading ID <code class="pf-inline-code">phase-1-heading</code>, output cell 5 ID <code class="pf-inline-code">phase-1-orchestration</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="pf-inline-code">render_phase_summary(1, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/environment/phase_1_signoff.json">phase_1_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/utils/environment.py</code></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Input</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../requirements.txt"><code class="pf-inline-code">requirements.txt</code></a>, <a class="pf-link pf-artifact-link" href="../../pyproject.toml"><code class="pf-inline-code">pyproject.toml</code></a></td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/environment/environment_report.json"><code class="pf-inline-code">artifacts/environment/environment_report.json</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/environment/phase_1_signoff.json"><code class="pf-inline-code">phase_1_signoff.json</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/environment/requirements_freeze.txt"><code class="pf-inline-code">requirements_freeze.txt</code></a></td>
</tr>
<tr>
<td><strong>Invariants</strong></td>
<td>Deterministic mode D0, MPS visibility drift controlled</td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Capture Python, torch, sklearn, numpy versions</li>
<li>Freeze requirements</li>
<li>Smoke test imports</li>
<li>Set deterministic seed</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-foundation" data-phase="2"><h2 class="pf-heading pf-h2" id="phase-2---data-acquisition"><span aria-hidden="true" class="pf-phase-pill">PHASE 2</span>Phase 2 - Data Acquisition</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 7–8</a> — heading ID <code class="pf-inline-code">phase-2-heading</code>, output cell 8 ID <code class="pf-inline-code">phase-2-orchestration</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="pf-inline-code">render_frozen_phase_evidence(2, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/acquisition/phase_2_signoff.json">phase_2_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/data/acquisition.py</code></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Input</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../data/raw_data/source/appliances_energy_prediction.zip"><code class="pf-inline-code">data/raw_data/source/appliances_energy_prediction.zip</code></a></td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><code class="pf-inline-code">artifacts/acquisition/dataset_manifest.json</code> <span class="pf-meta-chip">UNAVAILABLE IN CURRENT TREE</span>, <a class="pf-link pf-artifact-link" href="../../artifacts/acquisition/phase_2_signoff.json"><code class="pf-inline-code">phase_2_signoff.json</code></a></td>
</tr>
<tr>
<td><strong>Output (data)</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../data/raw_data/energydata_complete.csv"><code class="pf-inline-code">data/raw_data/energydata_complete.csv</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Verify SHA256 checksums</li>
<li>Materialize raw CSV</li>
<li>Generate dataset manifest</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-foundation" data-phase="3"><h2 class="pf-heading pf-h2" id="phase-3---schema-audit"><span aria-hidden="true" class="pf-phase-pill">PHASE 3</span>Phase 3 - Schema Audit</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 9–10</a> — heading ID <code class="pf-inline-code">phase-3-heading</code>, output cell 10 ID <code class="pf-inline-code">phase-3-orchestration</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="pf-inline-code">render_phase_summary(3, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/schema/phase_3_signoff.json">phase_3_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/data/schema.py</code></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Input</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../data/raw_data/energydata_complete.csv"><code class="pf-inline-code">data/raw_data/energydata_complete.csv</code></a></td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/schema/schema_manifest.json"><code class="pf-inline-code">artifacts/schema/schema_manifest.json</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/schema/schema_summary.csv"><code class="pf-inline-code">schema_summary.csv</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/schema/variable_dictionary.csv"><code class="pf-inline-code">variable_dictionary.csv</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/schema/phase_3_signoff.json"><code class="pf-inline-code">phase_3_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Validate dtypes</li>
<li>Detect schema drift</li>
<li>Generate variable dictionary</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-foundation" data-phase="4"><h2 class="pf-heading pf-h2" id="phase-4---temporal-integrity-audit"><span aria-hidden="true" class="pf-phase-pill">PHASE 4</span>Phase 4 - Temporal Integrity Audit</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 11–12</a> — heading ID <code class="pf-inline-code">phase-4-heading</code>, output cell 12 ID <code class="pf-inline-code">phase-4-orchestration</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="pf-inline-code">render_phase_summary(4, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/temporal/phase_4_signoff.json">phase_4_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/data/temporal.py</code></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Input</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/schema/"><code class="pf-inline-code">artifacts/schema/</code></a></td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/temporal/temporal_manifest.json"><code class="pf-inline-code">artifacts/temporal/temporal_manifest.json</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/temporal/daily_observation_counts.csv"><code class="pf-inline-code">daily_observation_counts.csv</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/temporal/interval_distribution.csv"><code class="pf-inline-code">interval_distribution.csv</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/temporal/phase_4_signoff.json"><code class="pf-inline-code">phase_4_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Verify cadence = 10 minutes</li>
<li>Detect gaps, duplicates</li>
<li>Continuity segments</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-foundation" data-phase="5"><h2 class="pf-heading pf-h2" id="phase-5---chronological-split"><span aria-hidden="true" class="pf-phase-pill">PHASE 5</span>Phase 5 - Chronological Split</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 13–14</a> — heading ID <code class="pf-inline-code">phase-5-heading</code>, output cell 14 ID <code class="pf-inline-code">phase-5-orchestration</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="pf-inline-code">render_phase_summary(5, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/splits/phase_5_signoff.json">phase_5_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <a class="pf-link" href="../../src/course_work/data/splitting.py">splitting.py</a></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Input</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/temporal/"><code class="pf-inline-code">artifacts/temporal/</code></a>, canonical raw dataset và base contract</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/splits/"><code class="pf-inline-code">artifacts/splits/</code></a><code class="pf-inline-code">{split_manifest.json, split_membership.csv, split_boundaries.csv, split_summary.csv}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/splits/phase_5_signoff.json"><code class="pf-inline-code">phase_5_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Tạo chronological Train / Validation / Test split theo contract.</li>
<li>Kiểm tra leakage và boundary neighborhood.</li>
<li>Khóa split membership/fingerprint cho downstream phases.</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-foundation" data-phase="6"><h2 class="pf-heading pf-h2" id="phase-6---exploratory-data-analysis"><span aria-hidden="true" class="pf-phase-pill">PHASE 6</span>Phase 6 - Exploratory Data Analysis</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 15–44</a> — heading ID <code class="pf-inline-code">phase-6-heading</code>, output cell 16 ID <code class="pf-inline-code">phase-6-setup</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="pf-inline-code">render_phase_summary(6, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/eda/phase_6_signoff.json">phase_6_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <a class="pf-link" href="../../src/course_work/data/eda.py">eda.py</a> và <a class="pf-link" href="../../src/course_work/reporting/eda.py">reporting/eda.py</a></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Input</strong></td>
<td>Raw dataset cùng verified Phase 2–5 signoffs/split context</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/eda/"><code class="pf-inline-code">artifacts/eda/</code></a><code class="pf-inline-code">{figures,tables,eda_manifest.json,eda_anomalies.json}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/eda/phase_6_signoff.json"><code class="pf-inline-code">phase_6_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Target distribution, ECDF, timeline và representative-week views.</li>
<li>Hourly/weekday profiles, feature distributions và numerical relationships.</li>
<li>Cross-correlation, IQR diagnostics, smoothing demonstration và time-series diagnostics.</li>
<li>Trình bày 14 EDA sub-sections tại Cells 17–44.</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-foundation" data-phase="7"><h2 class="pf-heading pf-h2" id="phase-7---feature-engineering"><span aria-hidden="true" class="pf-phase-pill">PHASE 7</span>Phase 7 - Feature Engineering</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 45–46</a> — heading ID <code class="pf-inline-code">phase-7-heading</code>, output cell 46 ID <code class="pf-inline-code">phase-7-orchestration</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="pf-inline-code">render_phase_summary(7, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/features/phase_7_signoff.json">phase_7_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <a class="pf-link" href="../../src/course_work/data/features.py">features.py</a></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Input</strong></td>
<td>Verified EDA manifest, Phase 5 split signoff và upstream contracts</td>
</tr>
<tr>
<td><strong>Output (data)</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../data/interim/uci_appliances_energy_prediction/energydata_feature_engineered_v1.csv"><code class="pf-inline-code">data/interim/uci_appliances_energy_prediction/energydata_feature_engineered_v1.csv</code></a></td>
</tr>
<tr>
<td><strong>Output (artifacts)</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/features/"><code class="pf-inline-code">artifacts/features/</code></a><code class="pf-inline-code">{feature_engineering_manifest.json, feature_registry.csv, feature_lineage.csv}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/features/phase_7_signoff.json"><code class="pf-inline-code">phase_7_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Tạo time/lag/rolling features theo contract.</li>
<li>Giữ feature lineage và leakage audit.</li>
<li>Khóa checksum cho feature-engineered dataset.</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-foundation" data-phase="8"><h2 class="pf-heading pf-h2" id="phase-8---feature-set-variants"><span aria-hidden="true" class="pf-phase-pill">PHASE 8</span>Phase 8 - Feature-Set Variants</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 47–48</a> — heading ID <code class="pf-inline-code">phase-8-heading</code>, output cell 48 ID <code class="pf-inline-code">phase-8-orchestration</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="pf-inline-code">render_phase_summary(8, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/feature_sets/phase_8_signoff.json">phase_8_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <a class="pf-link" href="../../src/course_work/data/feature_sets.py">feature_sets.py</a></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Input</strong></td>
<td>Verified Phase 7 feature artifacts và engineered dataset</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/feature_sets/"><code class="pf-inline-code">artifacts/feature_sets/</code></a><code class="pf-inline-code">{feature_set_registry.json, feature_components.json, feature_order_checks.csv, feature_set_lineage.csv}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/feature_sets/phase_8_signoff.json"><code class="pf-inline-code">phase_8_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Định nghĩa FS0/FS1/FS2 và ordered feature lists.</li>
<li>Theo dõi lineage/fingerprint cho từng variant.</li>
<li>Kiểm tra feature order và leakage.</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-foundation" data-phase="9"><h2 class="pf-heading pf-h2" id="phase-9---train-only-scaling"><span aria-hidden="true" class="pf-phase-pill">PHASE 9</span>Phase 9 - Train-Only Scaling</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 49–50</a> — heading ID <code class="pf-inline-code">phase-9-heading</code>, output cell 50 ID <code class="pf-inline-code">phase-9-orchestration</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="pf-inline-code">render_phase_summary(9, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/scaling/phase_9_signoff.json">phase_9_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/data/scaling.py</code> (also re-exported at <code class="pf-inline-code">course_work.scaling</code>)</p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Input</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/splits/"><code class="pf-inline-code">artifacts/splits/</code></a></td>
</tr>
<tr>
<td><strong>Output (scalers)</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/scalers/"><code class="pf-inline-code">artifacts/scalers/</code></a><code class="pf-inline-code">{x,y}/*.joblib</code></td>
</tr>
<tr>
<td><strong>Output (artifacts)</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/scaling/"><code class="pf-inline-code">artifacts/scaling/</code></a><code class="pf-inline-code">{scaling_manifest.json, scaler_registry.json}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/scaling/phase_9_signoff.json"><code class="pf-inline-code">phase_9_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Fit scalers on Train only</li>
<li>Scale Validation and Test using Train statistics</li>
<li>SHA256 checksums</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-foundation" data-phase="10"><h2 class="pf-heading pf-h2" id="phase-10---window-builder"><span aria-hidden="true" class="pf-phase-pill">PHASE 10</span>Phase 10 - Window Builder</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 51–52</a> — heading ID <code class="pf-inline-code">phase-10-heading</code>, output cell 52 ID <code class="pf-inline-code">phase-10-orchestration</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="pf-inline-code">render_phase_summary(10, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/windows/phase_10_signoff.json">phase_10_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/data/windows.py</code></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Input</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/scaling/"><code class="pf-inline-code">artifacts/scaling/</code></a></td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/windows/window_manifest.json"><code class="pf-inline-code">artifacts/windows/window_manifest.json</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/windows/window_population_summary.csv"><code class="pf-inline-code">window_population_summary.csv</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/windows/phase_10_signoff.json"><code class="pf-inline-code">phase_10_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Build lookback windows</li>
<li>Compute population summaries</li>
<li>Leakage audit</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-modeling" data-phase="11"><h2 class="pf-heading pf-h2" id="phase-11---dataloaders"><span aria-hidden="true" class="pf-phase-pill">PHASE 11</span>Phase 11 - DataLoaders</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 53–54</a> — heading ID <code class="pf-inline-code">phase-11-heading</code>, output cell 54 ID <code class="pf-inline-code">phase-11-orchestration</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="pf-inline-code">render_phase_summary(11, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/dataloaders/phase_11_signoff.json">phase_11_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/data/datasets.py</code></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Input</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/windows/"><code class="pf-inline-code">artifacts/windows/</code></a></td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/dataloaders/"><code class="pf-inline-code">artifacts/dataloaders/</code></a><code class="pf-inline-code">{dataloader_manifest.json, dataloader_registry.csv}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/dataloaders/phase_11_signoff.json"><code class="pf-inline-code">phase_11_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Train / Validation / Test DataLoaders</li>
<li>Sequential ordering preserved</li>
<li>Shuffle reproducibility audit</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-modeling" data-phase="12"><h2 class="pf-heading pf-h2" id="phase-12---shared-metrics"><span aria-hidden="true" class="pf-phase-pill">PHASE 12</span>Phase 12 - Shared Metrics</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 55–56</a> — heading ID <code class="pf-inline-code">phase-12-heading</code>, output cell 56 ID <code class="pf-inline-code">phase-12-orchestration</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="pf-inline-code">render_phase_summary(12, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/metrics/phase_12_signoff.json">phase_12_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/evaluation/metrics.py</code></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Input</strong></td>
<td>None (pure functions)</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/metrics/"><code class="pf-inline-code">artifacts/metrics/</code></a><code class="pf-inline-code">{metric_manifest.json, metric_unit_tests.csv}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/metrics/phase_12_signoff.json"><code class="pf-inline-code">phase_12_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>MAE, RMSE, R² in original Wh units</li>
<li>Reference examples</li>
<li>Test firewall audit</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-modeling" data-phase="13"><h2 class="pf-heading pf-h2" id="phase-13---experiment-registry"><span aria-hidden="true" class="pf-phase-pill">PHASE 13</span>Phase 13 - Experiment Registry</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 57–58</a> — heading ID <code class="pf-inline-code">phase-13-heading</code>, output cell 58 ID <code class="pf-inline-code">phase-13-orchestration</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="pf-inline-code">render_phase_summary(13, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/experiments/phase_13_signoff.json">phase_13_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/experiments/registry.py</code></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Input</strong></td>
<td>All upstream artifacts</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/experiments/"><code class="pf-inline-code">artifacts/experiments/</code></a><code class="pf-inline-code">{experiment_registry.jsonl, run_artifact_registry.csv}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/experiments/phase_13_signoff.json"><code class="pf-inline-code">phase_13_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Register every training run</li>
<li>Track configs, metrics, status</li>
<li>Family taxonomy</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-modeling" data-phase="14"><h2 class="pf-heading pf-h2" id="phase-14---persistence-baseline"><span aria-hidden="true" class="pf-phase-pill">PHASE 14</span>Phase 14 - Persistence Baseline</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 59–60</a> — heading ID <code class="pf-inline-code">phase-14-heading</code>, output cell 60 ID <code class="pf-inline-code">phase-14-orchestration</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="pf-inline-code">render_phase_summary(14, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/baselines/persistence/phase_14_signoff.json">phase_14_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/baselines/persistence.py</code></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Input</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/dataloaders/"><code class="pf-inline-code">artifacts/dataloaders/</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/scaling/"><code class="pf-inline-code">artifacts/scaling/</code></a></td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/baselines/persistence/"><code class="pf-inline-code">artifacts/baselines/persistence/</code></a><code class="pf-inline-code">{persistence_manifest.json, persistence_validation_metrics.json}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/baselines/persistence/phase_14_signoff.json"><code class="pf-inline-code">phase_14_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Persistence forecast: $\hat{y}[t+1] = y[t]$</li>
<li>Compare with LSTM and Transformer baselines later</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-modeling" data-phase="15"><h2 class="pf-heading pf-h2" id="phase-15---lstm-implementation"><span aria-hidden="true" class="pf-phase-pill">PHASE 15</span>Phase 15 - LSTM Implementation</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 61–62</a> — heading ID <code class="pf-inline-code">1d1eebe5</code>, output cell 62 ID <code class="pf-inline-code">499bc911</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="pf-inline-code">render_phase_summary(15, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/models/lstm/phase_15_signoff.json">phase_15_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/models/lstm_regressor.py</code></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Input</strong></td>
<td>None (architecture code)</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/models/lstm/"><code class="pf-inline-code">artifacts/models/lstm/</code></a><code class="pf-inline-code">{lstm_model_manifest.json, lstm_shape_contract.json, lstm_unit_tests.csv}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/models/lstm/phase_15_signoff.json"><code class="pf-inline-code">phase_15_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>LSTM regressor (input → hidden → output)</li>
<li>Shape contracts</li>
<li>Parameter audit</li>
<li>Unit tests</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-modeling" data-phase="16"><h2 class="pf-heading pf-h2" id="phase-16---transformer-implementation"><span aria-hidden="true" class="pf-phase-pill">PHASE 16</span>Phase 16 - Transformer Implementation</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 63–64</a> — heading ID <code class="pf-inline-code">d91e6b47</code>, output cell 64 ID <code class="pf-inline-code">518fca61</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="pf-inline-code">render_phase_summary(16, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/models/transformer/phase_16_signoff.json">phase_16_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/models/transformer_regressor.py</code> + <code class="pf-inline-code">transformer_encoder_layer.py</code> + <code class="pf-inline-code">positional_encoding.py</code></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Input</strong></td>
<td>None (architecture code)</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/models/transformer/"><code class="pf-inline-code">artifacts/models/transformer/</code></a><code class="pf-inline-code">{transformer_model_manifest.json, transformer_shape_contract.json, ...}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/models/transformer/phase_16_signoff.json"><code class="pf-inline-code">phase_16_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Transformer encoder regressor</li>
<li>Positional encoding (sin/cos)</li>
<li>Multi-head self-attention</li>
<li>Shape + attention contracts</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-modeling" data-phase="17"><h2 class="pf-heading pf-h2" id="phase-17---attention-aware-encoder-verification"><span aria-hidden="true" class="pf-phase-pill">PHASE 17</span>Phase 17 - Attention-Aware Encoder Verification</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 65–66</a> — heading ID <code class="pf-inline-code">8906dc9f</code>, output cell 66 ID <code class="pf-inline-code">efc7f487</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="pf-inline-code">render_phase_summary(17, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/attention_verification/phase_17_signoff.json">phase_17_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/attention/verification.py</code></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Input</strong></td>
<td>Phase 16 outputs</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/attention_verification/"><code class="pf-inline-code">artifacts/attention_verification/</code></a><code class="pf-inline-code">{attention_verification_manifest.json, attention_path_equivalence_audit.csv, attention_probability_audit.csv, ...}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/attention_verification/phase_17_signoff.json"><code class="pf-inline-code">phase_17_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Verify attention weights sum to 1 per query</li>
<li>Probability/mask audit</li>
<li>Aggregation audit</li>
<li>Path equivalence (manual vs framework)</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-modeling" data-phase="18"><h2 class="pf-heading pf-h2" id="phase-18---forward-pass-sanity-tests"><span aria-hidden="true" class="pf-phase-pill">PHASE 18</span>Phase 18 - Forward-Pass Sanity Tests</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 67–68</a> — heading ID <code class="pf-inline-code">6dee6a62</code>, output cell 68 ID <code class="pf-inline-code">d60e1951</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="pf-inline-code">render_phase_summary(18, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/forward_sanity/phase_18_signoff.json">phase_18_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/sanity/forward_sanity.py</code></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Input</strong></td>
<td>Phase 16 + 17</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/forward_sanity/"><code class="pf-inline-code">artifacts/forward_sanity/</code></a><code class="pf-inline-code">{forward_sanity_manifest.json, forward_batch_audit.csv, ...}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/forward_sanity/phase_18_signoff.json"><code class="pf-inline-code">phase_18_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Test forward pass correctness</li>
<li>Batch independence</li>
<li>Parameter mutation audit</li>
<li>Device transfer</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-modeling" data-phase="19"><h2 class="pf-heading pf-h2" id="phase-19---baseline-training-engine"><span aria-hidden="true" class="pf-phase-pill">PHASE 19</span>Phase 19 - Baseline Training Engine</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 69–70</a> — heading ID <code class="pf-inline-code">a5fd715f</code>, output cell 70 ID <code class="pf-inline-code">2acc38f9</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="pf-inline-code">render_phase_summary(19, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/training_engine/phase_19_signoff.json">phase_19_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/training/engine.py</code> + <code class="pf-inline-code">losses.py</code></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Input</strong></td>
<td>Phase 11, 15, 16</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/training_engine/"><code class="pf-inline-code">artifacts/training_engine/</code></a><code class="pf-inline-code">{training_engine_manifest.json, training_engine_unit_tests.csv, ...}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/training_engine/phase_19_signoff.json"><code class="pf-inline-code">phase_19_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Generic training loop</li>
<li>Checkpoint save/load (best + last)</li>
<li>Early stopping</li>
<li>Gradient clipping</li>
<li>Resume support</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-modeling" data-phase="20"><h2 class="pf-heading pf-h2" id="phase-20---lstm-baseline-run"><span aria-hidden="true" class="pf-phase-pill">PHASE 20</span>Phase 20 - LSTM Baseline Run</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 71–72</a> — heading ID <code class="pf-inline-code">11dc2d67</code>, output cell 72 ID <code class="pf-inline-code">2eb13c9c</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="pf-inline-code">render_phase_summary(20, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/lstm_baseline/phase_20_signoff.json">phase_20_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/baselines/lstm_baseline.py</code></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Input</strong></td>
<td>Phase 19 (training engine)</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><code class="pf-inline-code">artifacts/baselines/lstm_baseline/{lstm_baseline_run_contract.json, lstm_baseline_discrepancies.json, ...}</code> <span class="pf-meta-chip">UNAVAILABLE IN CURRENT TREE</span>, <a class="pf-link pf-artifact-link" href="../../artifacts/lstm_baseline/phase_20_signoff.json"><code class="pf-inline-code">phase_20_signoff.json</code></a></td>
</tr>
<tr>
<td><strong>Output (runs)</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/runs/"><code class="pf-inline-code">artifacts/runs/</code></a><code class="pf-inline-code">RUN_LS_LS_*</code></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Train reference LSTM</li>
<li>Validation metrics vs persistence</li>
<li>Compare to Transformer_B0</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-modeling" data-phase="21"><h2 class="pf-heading pf-h2" id="phase-21---transformer-b0-run"><span aria-hidden="true" class="pf-phase-pill">PHASE 21</span>Phase 21 - Transformer B0 Run</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 73–74</a> — heading ID <code class="pf-inline-code">328a4c95</code>, output cell 74 ID <code class="pf-inline-code">03db9e36</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="pf-inline-code">render_phase_summary(21, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/transformer_b0/phase_21_signoff.json">phase_21_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/baselines/transformer_b0.py</code></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Input</strong></td>
<td>Phase 19</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/transformer_b0/"><code class="pf-inline-code">artifacts/transformer_b0/</code></a><code class="pf-inline-code">{transformer_b0_run_contract.json, ...}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/transformer_b0/phase_21_signoff.json"><code class="pf-inline-code">phase_21_signoff.json</code></a></td>
</tr>
<tr>
<td><strong>Output (runs)</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/runs/"><code class="pf-inline-code">artifacts/runs/</code></a><code class="pf-inline-code">RUN_TR_B0_*</code></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Train reference Transformer B0</li>
<li>Population audit (same Train/Val/Test as LSTM baseline)</li>
<li>Validation metrics comparison</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-modeling" data-phase="22"><h2 class="pf-heading pf-h2" id="phase-22---learning-curve-diagnostics"><span aria-hidden="true" class="pf-phase-pill">PHASE 22</span>Phase 22 - Learning-Curve Diagnostics</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 75–76</a> — heading ID <code class="pf-inline-code">410cc5e0</code>, output cell 76 ID <code class="pf-inline-code">cd4716af</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="pf-inline-code">render_frozen_phase_evidence(22, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/learning_diagnostics/phase_22_signoff.json">phase_22_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/diagnostics/learning_diagnostics.py</code></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Input</strong></td>
<td>Phase 20, 21 runs</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/learning_diagnostics/"><code class="pf-inline-code">artifacts/learning_diagnostics/</code></a><code class="pf-inline-code">{learning_diagnostics_manifest.json, learning_diagnostics_summary.csv}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/learning_diagnostics/phase_22_signoff.json"><code class="pf-inline-code">phase_22_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Train vs Validation loss curves</li>
<li>Overfit detection</li>
<li>Gradient norms</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-sweeps" data-phase="23"><h2 class="pf-heading pf-h2" id="phase-23---s1-feature-set-sweep"><span aria-hidden="true" class="pf-phase-pill">PHASE 23</span>Phase 23 - S1 Feature-Set Sweep</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 77–78</a> — heading ID <code class="pf-inline-code">08a52851</code>, output cell 78 ID <code class="pf-inline-code">0881cfe3</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="pf-inline-code">render_frozen_phase_evidence(23, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/sweeps/s1_feature_set/phase_23_signoff.json">phase_23_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/experiments/phase_execution.py</code></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Input</strong></td>
<td>Phase 7, 11, 19</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/s1_feature_set/"><code class="pf-inline-code">artifacts/sweeps/s1_feature_set/</code></a><code class="pf-inline-code">{sweep_manifest.json, results.csv, s1_feature_set_winner.json, s1_reference_update.json}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/s1_feature_set/phase_23_signoff.json"><code class="pf-inline-code">phase_23_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Sweep over FS0, FS1, FS2</li>
<li>Pick winner by validation RMSE</li>
<li>Update reference config</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-sweeps" data-phase="24"><h2 class="pf-heading pf-h2" id="phase-24---s2-time-feature-sweep"><span aria-hidden="true" class="pf-phase-pill">PHASE 24</span>Phase 24 - S2 Time-Feature Sweep</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 79–80</a> — heading ID <code class="pf-inline-code">35e8f752</code>, output cell 80 ID <code class="pf-inline-code">9b2d6e88</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="pf-inline-code">render_frozen_phase_evidence(24, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/sweeps/s2_time_feature/phase_24_signoff.json">phase_24_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/sweeps/</code> (via <code class="pf-inline-code">experiments/phase_execution.py</code>)</p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/s2_time_feature/"><code class="pf-inline-code">artifacts/sweeps/s2_time_feature/</code></a><code class="pf-inline-code">{..., s2_time_feature_winner.json, s2_reference_update.json}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/s2_time_feature/phase_24_signoff.json"><code class="pf-inline-code">phase_24_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Sweep time feature variants</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-sweeps" data-phase="25"><h2 class="pf-heading pf-h2" id="phase-25---s3-target-scaling-sweep"><span aria-hidden="true" class="pf-phase-pill">PHASE 25</span>Phase 25 - S3 Target-Scaling Sweep</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 81–82</a> — heading ID <code class="pf-inline-code">1a1d6b45</code>, output cell 82 ID <code class="pf-inline-code">51bc5965</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="pf-inline-code">render_frozen_phase_evidence(25, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/sweeps/s3_target_scaling/phase_25_signoff.json">phase_25_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/s3_target_scaling/"><code class="pf-inline-code">artifacts/sweeps/s3_target_scaling/</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/s3_target_scaling/phase_25_signoff.json"><code class="pf-inline-code">phase_25_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Sweep target scaling options</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-sweeps" data-phase="26"><h2 class="pf-heading pf-h2" id="phase-26---s4-lookback-sweep"><span aria-hidden="true" class="pf-phase-pill">PHASE 26</span>Phase 26 - S4 Lookback Sweep</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 83–84</a> — heading ID <code class="pf-inline-code">2ef1b3d9</code>, output cell 84 ID <code class="pf-inline-code">585b390d</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="pf-inline-code">render_frozen_phase_evidence(26, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/sweeps/s4_lookback/phase_26_signoff.json">phase_26_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/s4_lookback/"><code class="pf-inline-code">artifacts/sweeps/s4_lookback/</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/s4_lookback/phase_26_signoff.json"><code class="pf-inline-code">phase_26_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Sweep <code class="pf-inline-code">lookback ∈ {36, 72, 144}</code></li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-sweeps" data-phase="27"><h2 class="pf-heading pf-h2" id="phase-27---s5-pooling-sweep"><span aria-hidden="true" class="pf-phase-pill">PHASE 27</span>Phase 27 - S5 Pooling Sweep</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 85–86</a> — heading ID <code class="pf-inline-code">2b25f489</code>, output cell 86 ID <code class="pf-inline-code">86f4ac0c</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="pf-inline-code">render_frozen_phase_evidence(27, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/sweeps/s5_pooling/phase_27_signoff.json">phase_27_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/s5_pooling/"><code class="pf-inline-code">artifacts/sweeps/s5_pooling/</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/s5_pooling/phase_27_signoff.json"><code class="pf-inline-code">phase_27_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Sweep pooling strategies (last, mean, max, attention)</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-sweeps" data-phase="28"><h2 class="pf-heading pf-h2" id="phase-28---s6-activation-sweep"><span aria-hidden="true" class="pf-phase-pill">PHASE 28</span>Phase 28 - S6 Activation Sweep</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 87–88</a> — heading ID <code class="pf-inline-code">b020e51d</code>, output cell 88 ID <code class="pf-inline-code">b70c9707</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="pf-inline-code">render_frozen_phase_evidence(28, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/sweeps/s6_activation/phase_28_signoff.json">phase_28_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/s6_activation/"><code class="pf-inline-code">artifacts/sweeps/s6_activation/</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/s6_activation/phase_28_signoff.json"><code class="pf-inline-code">phase_28_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Sweep activation functions (ReLU, GELU, SiLU, ...)</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-sweeps" data-phase="29"><h2 class="pf-heading pf-h2" id="phase-29---s7-batch-size-sweep"><span aria-hidden="true" class="pf-phase-pill">PHASE 29</span>Phase 29 - S7 Batch-Size Sweep</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 89–90</a> — heading ID <code class="pf-inline-code">aa9fb435</code>, output cell 90 ID <code class="pf-inline-code">eb4f1b80</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="pf-inline-code">render_frozen_phase_evidence(29, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/sweeps/s7_batch_size/phase_29_signoff.json">phase_29_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/s7_batch_size/"><code class="pf-inline-code">artifacts/sweeps/s7_batch_size/</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/s7_batch_size/phase_29_signoff.json"><code class="pf-inline-code">phase_29_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Sweep batch sizes</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-sweeps" data-phase="30"><h2 class="pf-heading pf-h2" id="phase-30---s8-learning-rate-sweep"><span aria-hidden="true" class="pf-phase-pill">PHASE 30</span>Phase 30 - S8 Learning-Rate Sweep</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 91–92</a> — heading ID <code class="pf-inline-code">4a8618de</code>, output cell 92 ID <code class="pf-inline-code">3fe4478c</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="pf-inline-code">render_frozen_phase_evidence(30, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/sweeps/s8_learning_rate/phase_30_signoff.json">phase_30_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/s8_learning_rate/"><code class="pf-inline-code">artifacts/sweeps/s8_learning_rate/</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/s8_learning_rate/phase_30_signoff.json"><code class="pf-inline-code">phase_30_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Sweep learning rate values</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-sweeps" data-phase="31"><h2 class="pf-heading pf-h2" id="phase-31---s9-weight-decay-sweep"><span aria-hidden="true" class="pf-phase-pill">PHASE 31</span>Phase 31 - S9 Weight-Decay Sweep</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 93–94</a> — heading ID <code class="pf-inline-code">phase-31-heading</code>, output cell 94 ID <code class="pf-inline-code">phase-31-resume</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="pf-inline-code">render_frozen_phase_evidence(31, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/sweeps/S9_weight_decay/phase_31_signoff.json">phase_31_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/sweeps/weight_decay.py</code></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S9_weight_decay/"><code class="pf-inline-code">artifacts/sweeps/S9_weight_decay/</code></a><code class="pf-inline-code">{sweep_manifest.json, s9_weight_decay_metrics.csv, s9_weight_decay_winner.json, s9_reference_update.json}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S9_weight_decay/phase_31_signoff.json"><code class="pf-inline-code">phase_31_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Sweep weight decay values</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-sweeps" data-phase="32"><h2 class="pf-heading pf-h2" id="phase-32---s10-dropout-sweep"><span aria-hidden="true" class="pf-phase-pill">PHASE 32</span>Phase 32 - S10 Dropout Sweep</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 95–96</a> — heading ID <code class="pf-inline-code">phase-32-heading</code>, output cell 96 ID <code class="pf-inline-code">phase-32-resume</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="pf-inline-code">render_frozen_phase_evidence(32, PROJECT_ROOT)</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/sweeps/S10_dropout/phase_32_signoff.json">phase_32_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/sweeps/dropout.py</code></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S10_dropout/"><code class="pf-inline-code">artifacts/sweeps/S10_dropout/</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S10_dropout/phase_32_signoff.json"><code class="pf-inline-code">phase_32_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Sweep dropout rates</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-group-sweeps" data-phase="33"><h2 class="pf-heading pf-h2" id="transformer-configuration-after-phase-33"><span aria-hidden="true" class="pf-phase-pill">PHASE 33</span>Transformer Configuration after Phase 33</h2>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: <a class="pf-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 97–98</a> — heading ID <code class="pf-inline-code">phase-33-config-heading</code>, output cell 98 ID <code class="pf-inline-code">phase-33-config-display</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="pf-inline-code">render_phase_33_transformer_configuration()</code>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/sweeps/S11_d_model/phase_33_signoff.json">phase_33_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code">V1</code>; notebook presentation is read-only.</li>
</ul>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/sweeps/d_model.py</code></p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Chi tiết</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S11_d_model/"><code class="pf-inline-code">artifacts/sweeps/S11_d_model/</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S11_d_model/phase_33_signoff.json"><code class="pf-inline-code">phase_33_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Sweep d_model (transformer hidden dimension)</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-summary-card"><h2 class="pf-heading pf-h2" id="tổng-kết-phases-133">Tổng Kết Phases 1–33</h2>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Nhóm</th>
<th style="text-align:right">Số phases</th>
<th>Đặc điểm</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Foundation</strong> (1–10)</td>
<td style="text-align:right">10</td>
<td>Environment → acquisition → schema → temporal → split → EDA → features → feature sets → scaling → windows</td>
</tr>
<tr>
<td><strong>Modeling</strong> (11–22)</td>
<td style="text-align:right">12</td>
<td>DataLoaders, metrics, registry, models, baselines, training engine và diagnostics</td>
</tr>
<tr>
<td><strong>Sweeps 1</strong> (23–33)</td>
<td style="text-align:right">11</td>
<td>S1–S11 validation sweeps; notebook chỉ render verified/frozen evidence</td>
</tr>
</tbody>
</table></div>
<p>Các sweep giữ pattern khoa học lịch sử:</p>
<ol>
<li>Đọc previous winner/reference.</li>
<li>Chạy candidate matrix theo contract lịch sử.</li>
<li>Chọn winner bằng Validation evidence.</li>
<li>Cập nhật reference cho phase kế tiếp.</li>
</ol>
<p>Current notebook không tự chạy lại các sweep khi render.</p>
<p><strong>Handoff:</strong> Phase 33 khóa Transformer configuration tại <a class="pf-link" href="../../artifacts/sweeps/S11_d_model/phase_33_signoff.json">phase_33_signoff.json</a>. Xem <a class="pf-link" href="PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md">PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md</a>.</p>
<p><strong>Phiên bản:</strong> 16/09/2026 — baseline chi tiết được giữ và đồng bộ với current verified flow.</p>
</section></div>
<a class="pf-handoff-link" href="PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md" aria-label="Open PHASE 34 TO 59 Current Flow Detail">
<div class="pf-handoff">
  <div class="pf-handoff-grid">
    <div class="pf-handoff-icon"><svg viewBox="0 0 24 24" width="26" height="26"><path d="M4 12h14M14 7l5 5-5 5" fill="none" stroke="#0b63ce" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
    <div><h3>Handoff · Phase 33 → Phase 34–59</h3><p>Transformer configuration được khóa tại Phase 33; flow tiếp tục sang tài liệu PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md.</p></div>
    <div class="pf-handoff-arrow">→</div>
  </div>
</div>
</a>
<div class="pf-footer">
<h3>End of Phase 1–33 · Verified V1 Flow</h3>
<p>Environment → Data → Features → Windows → Models → Training → Diagnostics → Validation Sweeps</p>
<div class="pf-floaters">
<span class="pf-floater"><svg viewBox="0 0 24 24" aria-hidden="true"><path fill="#3776AB" d="M12 2c-4 0-4 2-4 4v2h8v1H6c-2 0-4 1-4 4s2 4 4 4h2v-3c0-2 2-4 4-4h6c2 0 4-2 4-4s-2-4-4-4z"/><circle cx="10" cy="5" r="1" fill="#fff"/><path fill="#FFD43B" d="M12 22c4 0 4-2 4-4v-2H8v-1h10c2 0 4-1 4-4s-2-4-4-4h-2v3c0 2-2 4-4 4H6c-2 0-4 2-4 4s2 4 4 4z"/><circle cx="14" cy="19" r="1" fill="#fff"/></svg></span>
<span class="pf-floater"><svg viewBox="0 0 24 24"><path d="M3 18h18M5 15l4-5 3 3 5-7 2 3" fill="none" stroke="#0b63ce" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></span>
<span class="pf-floater"><svg viewBox="0 0 24 24"><path d="M13 2 5 13h6l-1 9 9-13h-6z" fill="#f5c84b" stroke="#9a6700" stroke-width=".7"/></svg></span>
<span class="pf-floater"><svg viewBox="0 0 24 24"><path d="M5 5h14v14H5z" fill="none" stroke="#0b63ce" stroke-width="1.7"/><path d="M8 14l2-3 3 2 3-5" fill="none" stroke="#4cc9f0" stroke-width="1.8"/></svg></span>
<span class="pf-floater"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" fill="#e8fff7" stroke="#10b981" stroke-width="1.5"/><path d="m8 12 2.5 2.5L16.5 8.5" fill="none" stroke="#10b981" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg></span>
</div>
<svg class="pf-wave" viewBox="0 0 1200 120" preserveAspectRatio="none" aria-hidden="true"><path d="M0,58 C175,118 325,2 520,55 C720,112 920,12 1200,58 L1200,120 L0,120 Z" fill="#4cc9f0" opacity=".50"/><path d="M0,78 C215,26 385,116 610,66 C845,17 1018,102 1200,55 L1200,120 L0,120 Z" fill="#f5c84b" opacity=".48"/></svg>
</div>
</div>
