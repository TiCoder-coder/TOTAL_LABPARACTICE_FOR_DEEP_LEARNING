<!-- Refactored visual Markdown. CSS/SVG only; no JavaScript. Scientific content preserved from source. -->
<style>
:root{--pf-ink:#101828;--pf-muted:#475467;--pf-blue:#0b63ce;--pf-blue-2:#2f80ed;--pf-cyan:#4cc9f0;--pf-yellow:#f5c84b;--pf-yellow-soft:#fff6c8;--pf-paper:#fff;--pf-soft:#f6faff;--pf-line:#d8e7f5;--pf-green:#10b981;--pf-green-soft:#eafbf4}
*{box-sizing:border-box}html{scroll-behavior:smooth}.pf-shell{max-width:1220px;margin:0 auto;padding:24px 20px 0;background:linear-gradient(180deg,#fff 0%,#f7fbff 45%,#fffdf4 100%);color:var(--pf-ink);font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;line-height:1.72}
.pf-hero{position:relative;isolation:isolate;overflow:hidden;border:1px solid #cfe2f7;border-radius:30px;padding:38px 38px 30px;background:radial-gradient(circle at 84% 12%,rgba(245,200,75,.32),transparent 23%),radial-gradient(circle at 10% 8%,rgba(76,201,240,.23),transparent 28%),linear-gradient(135deg,#fff 0%,#f3f9ff 58%,#fff9dc 100%);box-shadow:0 22px 65px rgba(15,81,145,.12)}.pf-hero:before,.pf-hero:after{content:"";position:absolute;border-radius:50%;z-index:-1}.pf-hero:before{width:230px;height:230px;right:-78px;bottom:-102px;border:1px solid rgba(11,99,206,.18);animation:pfPulse 5s ease-in-out infinite}.pf-hero:after{width:115px;height:115px;right:-14px;bottom:-36px;border:1px solid rgba(245,200,75,.58);animation:pfPulse 4s ease-in-out infinite reverse}.pf-kicker{font-size:12px;font-weight:850;letter-spacing:.18em;text-transform:uppercase;color:#0758b3}.pf-title{margin:7px 0 10px!important;color:#0b1220!important;font-size:clamp(32px,5vw,54px)!important;line-height:1.05!important;border:0!important}.pf-subtitle{max-width:960px;margin:0;color:#344054;font-size:16px}.pf-subtitle strong{color:#0b63ce}
.pf-badges{display:flex;flex-wrap:wrap;gap:10px;margin-top:22px}.pf-badge{display:inline-flex;align-items:center;gap:8px;padding:8px 12px;border:1px solid #d8e7f6;border-radius:999px;background:rgba(255,255,255,.9);font-size:13px;font-weight:750;box-shadow:0 6px 18px rgba(15,81,145,.08);transition:transform .22s ease,box-shadow .22s ease}.pf-badge:hover{transform:translateY(-3px);box-shadow:0 10px 24px rgba(15,81,145,.14)}.pf-badge svg{width:19px;height:19px;display:block}
.pf-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:22px}.pf-stat{padding:14px 15px;border-radius:16px;background:rgba(255,255,255,.82);border:1px solid #dce9f6;backdrop-filter:blur(4px)}.pf-stat-value{font-size:22px;font-weight:900;line-height:1.05;color:#0b63ce}.pf-stat:nth-child(4) .pf-stat-value{color:#9a6800}.pf-stat-label{margin-top:6px;font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#667085}
.pf-flow{display:grid;grid-template-columns:1fr auto 1fr auto 1fr auto 1fr auto 1fr;gap:8px;align-items:center;margin:22px 0 6px}.pf-flow-node{padding:13px 11px;border-radius:14px;border:1px solid #d8e6f4;background:#fff;font-weight:800;text-align:center;box-shadow:0 5px 16px rgba(15,81,145,.06)}.pf-flow-node small{display:block;margin-top:2px;color:#667085;font-weight:650}.pf-flow-arrow{font-size:20px;color:#0b63ce;animation:pfArrow 1.8s ease-in-out infinite}
.pf-document{padding:10px 0 40px}.pf-section{position:relative;margin:20px 0;padding:25px 27px;background:rgba(255,255,255,.96);border:1px solid var(--pf-line);border-radius:21px;box-shadow:0 8px 28px rgba(15,81,145,.065);transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease}.pf-section:hover{transform:translateY(-2px);box-shadow:0 15px 36px rgba(15,81,145,.105);border-color:#bdd9f3}.pf-phase-card:before{content:"";position:absolute;left:0;top:20px;bottom:20px;width:5px;border-radius:0 6px 6px 0}.pf-lineage-recovered:before{background:linear-gradient(180deg,#0b63ce,#4cc9f0)}.pf-lineage-historical:before{background:linear-gradient(180deg,#f5c84b,#f59e0b)}.pf-lineage-v2final:before,.pf-lineage-v2closure:before{background:linear-gradient(180deg,#0b63ce,#10b981)}.pf-lineage-attention:before{background:linear-gradient(180deg,#f5c84b,#2f80ed)}
.pf-heading{scroll-margin-top:20px;color:#101828}.pf-h1{display:none}.pf-h2{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:0 0 14px!important;padding:0 0 12px;border-bottom:3px solid transparent;border-image:linear-gradient(90deg,var(--pf-blue),var(--pf-cyan),var(--pf-yellow),transparent) 1;font-size:27px!important;line-height:1.25!important}.pf-h3{margin:25px 0 10px!important;color:#0b5eb8!important;font-size:20px!important}.pf-h3:before{content:"◆";margin-right:8px;color:var(--pf-yellow);font-size:.72em}.pf-phase-pill{display:inline-flex;align-items:center;padding:5px 8px;border-radius:8px;background:#eaf4ff;color:#0758b3;font-size:10px;font-weight:900;letter-spacing:.1em;white-space:nowrap}.pf-lineage-historical .pf-phase-pill,.pf-lineage-attention .pf-phase-pill{background:#fff5c9;color:#855b00}.pf-lineage-v2final .pf-phase-pill,.pf-lineage-v2closure .pf-phase-pill{background:#eafbf4;color:#087a57}
.pf-phase-meta{display:flex;flex-wrap:wrap;gap:7px;margin:-2px 0 17px}.pf-meta-chip{display:inline-flex;align-items:center;padding:5px 9px;border-radius:999px;border:1px solid #d9e7f4;background:#f8fbff;color:#41556b;font-size:11px;font-weight:800;letter-spacing:.03em}.pf-meta-lineage{background:#eef6ff;color:#0758b3}.pf-lineage-historical .pf-meta-lineage,.pf-lineage-attention .pf-meta-lineage{background:#fff6cf;color:#7d5700;border-color:#f0dc86}.pf-lineage-v2final .pf-meta-lineage,.pf-lineage-v2closure .pf-meta-lineage{background:#eafbf4;color:#087a57;border-color:#bdebd9}.pf-meta-state{background:#edf6ff;color:#0758b3}
.pf-toc-card{background:linear-gradient(135deg,#fbfdff,#f3f9ff 68%,#fffaf0)}.pf-toc-card:after{content:"Navigation Map";position:absolute;top:18px;right:22px;padding:5px 9px;border-radius:999px;background:#fff3b8;color:#7c5700;font-size:10px;font-weight:900;letter-spacing:.08em;text-transform:uppercase}.pf-toc-list>li{margin:10px 0}.pf-toc-list ul{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px 18px;margin-top:8px}.pf-toc-list a{font-weight:650!important}.pf-lineage-card{background:radial-gradient(circle at 96% 8%,rgba(76,201,240,.18),transparent 27%),#fff}.pf-recovery-card{background:linear-gradient(135deg,#f9fcff,#eef7ff)}.pf-summary-card{background:radial-gradient(circle at 95% 10%,rgba(245,200,75,.20),transparent 28%),linear-gradient(135deg,#fff,#f4f9ff)}.pf-structure-card{background:linear-gradient(135deg,#fff,#f8fbff 60%,#fff9df)}
.pf-nav-notice{display:grid;grid-template-columns:auto 1fr;gap:14px;align-items:start;margin:18px 0;padding:17px 18px;border-radius:16px;border:1px solid #cfe2f7;background:linear-gradient(90deg,#edf6ff,#fffdf0);box-shadow:0 7px 20px rgba(15,81,145,.06)}.pf-nav-notice-icon{width:38px;height:38px;display:grid;place-items:center;border-radius:11px;background:#0b63ce;color:white;font-size:20px;font-weight:900}.pf-nav-notice strong{color:#0758b3}.pf-nav-notice p{margin:3px 0 0;color:#475467}.pf-notebook-link{display:inline-flex!important;align-items:center;padding:2px 8px;border-radius:999px;background:#eaf4ff;border:1px solid #cfe2f7;text-decoration:none!important;font-size:.92em}.pf-notebook-link:hover{background:#ddecff;text-decoration:none!important}
.pf-note{margin:18px 0!important;padding:16px 18px!important;border-left:4px solid var(--pf-blue)!important;background:linear-gradient(90deg,#edf6ff,#fffdf2)!important;border-radius:0 13px 13px 0;color:#26364a}.pf-rule{border:0!important;height:1px!important;background:linear-gradient(90deg,transparent,#93c5fd,#facc15,transparent)!important;margin:28px 0!important}.pf-link{color:#075fbe!important;text-decoration:none!important;font-weight:650}.pf-link:hover{text-decoration:underline!important;text-decoration-thickness:1.5px!important}.pf-inline-code{padding:.14em .42em;border:1px solid #d9e7f4;border-radius:6px;background:#f3f8fc;color:#854d0e;font-size:.92em}.pf-code-blue{background:#eaf4ff!important;color:#0758b3!important;border-color:#c9def4!important}.pf-code-yellow{background:#fff6c9!important;color:#7b5600!important;border-color:#f0dc86!important}.pf-code-green{background:#eafbf4!important;color:#087a57!important;border-color:#bdebd9!important}.pf-code{overflow:auto;padding:16px 18px;border:1px solid #cfe0ef;border-radius:14px;background:#f7fbff;box-shadow:inset 4px 0 0 #4cc9f0}.pf-code code{background:transparent!important;color:#111827!important}
.pf-table-wrap{overflow-x:auto;margin:18px 0;border:1px solid #d7e6f4;border-radius:14px;box-shadow:0 5px 18px rgba(15,81,145,.06)}.pf-table{width:100%;border-collapse:collapse;background:#fff;font-size:14px}.pf-table th{padding:11px 12px;background:linear-gradient(90deg,#eaf5ff,#fff6c9);color:#172033;text-align:left;border-bottom:1px solid #cbddeb}.pf-table td{padding:10px 12px;border-bottom:1px solid #edf2f7;vertical-align:top}.pf-table tr:last-child td{border-bottom:0}.pf-table tbody tr:hover{background:#f8fcff}ul,ol{padding-left:24px}li::marker{color:#0b63ce}strong{color:#111827}
.pf-handoff{position:relative;overflow:hidden;margin:24px 0 0;padding:23px 24px;border-radius:20px;border:1px solid #cfe2f7;background:linear-gradient(135deg,#eef7ff 0%,#fff 58%,#fff5c9 100%);box-shadow:0 10px 28px rgba(15,81,145,.08)}.pf-handoff-grid{display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:16px}.pf-handoff-icon{width:50px;height:50px;border-radius:15px;display:grid;place-items:center;background:#fff;border:1px solid #d7e6f4;box-shadow:0 6px 16px rgba(15,81,145,.08);color:#087a57;font-size:23px;font-weight:900}.pf-handoff h3{margin:0!important;color:#101828!important}.pf-handoff p{margin:3px 0 0;color:#475467}.pf-handoff-arrow{font-size:28px;color:#0b63ce;animation:pfArrow 1.8s ease-in-out infinite}
.pf-footer{position:relative;overflow:hidden;margin:24px -20px 0;padding:66px 24px 92px;text-align:center;background:linear-gradient(180deg,#f8fcff,#e9f6ff 54%,#fff4bf);border-top:1px solid #cfe4f7}.pf-footer h3{margin:0;color:#101828;font-size:24px}.pf-footer p{margin:6px 0;color:#475467}.pf-floaters{display:flex;justify-content:center;gap:17px;margin:22px 0 0}.pf-floater{width:38px;height:38px;display:grid;place-items:center;border-radius:12px;background:#fff;border:1px solid #d9e7f4;box-shadow:0 7px 18px rgba(15,81,145,.10);animation:pfFloat 3.2s ease-in-out infinite}.pf-floater:nth-child(2){animation-delay:.35s}.pf-floater:nth-child(3){animation-delay:.7s}.pf-floater:nth-child(4){animation-delay:1.05s}.pf-floater:nth-child(5){animation-delay:1.4s}.pf-floater svg{width:22px;height:22px}.pf-wave{position:absolute;left:-1%;right:-1%;bottom:-2px;width:102%;height:82px;opacity:.54}.pf-wave path:first-child{animation:pfWaveA 5.5s ease-in-out infinite alternate}.pf-wave path:last-child{animation:pfWaveB 6.5s ease-in-out infinite alternate}
@keyframes pfPulse{0%,100%{transform:scale(1);opacity:.65}50%{transform:scale(1.11);opacity:1}}@keyframes pfFloat{0%,100%{transform:translateY(0) rotate(0)}50%{transform:translateY(-11px) rotate(4deg)}}@keyframes pfArrow{0%,100%{transform:translateX(0);opacity:.65}50%{transform:translateX(5px);opacity:1}}@keyframes pfWaveA{from{transform:translateX(-9px)}to{transform:translateX(9px)}}@keyframes pfWaveB{from{transform:translateX(8px)}to{transform:translateX(-8px)}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important;scroll-behavior:auto!important}}@media(max-width:900px){.pf-flow{grid-template-columns:1fr}.pf-flow-arrow{transform:rotate(90deg)!important;text-align:center}}@media(max-width:760px){.pf-shell{padding:12px 10px 0}.pf-hero{padding:26px 20px;border-radius:21px}.pf-stats{grid-template-columns:repeat(2,minmax(0,1fr))}.pf-section{padding:19px 16px;border-radius:16px}.pf-h2{font-size:23px!important}.pf-toc-list ul{grid-template-columns:1fr}.pf-footer{margin-left:-10px;margin-right:-10px}.pf-handoff-grid{grid-template-columns:auto 1fr}.pf-handoff-arrow{display:none}.pf-nav-notice{grid-template-columns:1fr}.pf-nav-notice-icon{display:none}}
</style>
<div class="pf-shell">
<div class="pf-hero">
<div class="pf-kicker">CourseWork · Verified Current Flow</div>
<h1 class="pf-title">PHASE 34–59 · Current Flow Detail</h1>
<p class="pf-subtitle">Visual map of <strong>recovered V1 sweeps</strong>, <strong>historical V1 final pipeline</strong>, <strong>final V2 reporting</strong>, historical attention evidence, and V2 closure — with scientific lineage kept explicit.</p>
<div class="pf-badges">
<span class="pf-badge"><svg aria-hidden="true" viewbox="0 0 64 64"><path d="M31 5c-13 0-12 6-12 6v7h13v2H14S5 19 5 32s8 12 8 12h6v-8s0-8 8-8h13s7 0 7-7V12s1-7-16-7Zm-7 7a3 3 0 1 1 0 6 3 3 0 0 1 0-6Z" fill="#3776AB"></path><path d="M33 59c13 0 12-6 12-6v-7H32v-2h18s9 1 9-12-8-12-8-12h-6v8s0 8-8 8H24s-7 0-7 7v9s-1 7 16 7Zm7-7a3 3 0 1 1 0-6 3 3 0 0 1 0 6Z" fill="#FFD343"></path></svg><span>Python</span></span>
<span class="pf-badge"><span>🔥</span><span>PyTorch</span></span>
<span class="pf-badge"><span>◫</span><span>Jupyter</span></span>
<span class="pf-badge"><svg aria-hidden="true" viewbox="0 0 64 64"><circle cx="14" cy="16" fill="#0b63ce" r="5"></circle><circle cx="32" cy="10" fill="#f5c84b" r="5"></circle><circle cx="50" cy="16" fill="#0b63ce" r="5"></circle><circle cx="20" cy="34" fill="#4cc9f0" r="5"></circle><circle cx="44" cy="34" fill="#4cc9f0" r="5"></circle><circle cx="32" cy="52" fill="#f5c84b" r="5"></circle><path d="M18 18l10-6m8 0 10 6M17 20l2 9m28-9-2 9M24 35h16m-17 4 6 9m12-9-6 9" fill="none" stroke="#334155" stroke-linecap="round" stroke-width="3"></path></svg><span>Transformer</span></span>
<span class="pf-badge"><svg aria-hidden="true" viewbox="0 0 64 64"><path d="M9 52V12M9 52h46" stroke="#334155" stroke-linecap="round" stroke-width="4"></path><path d="m14 43 10-11 9 6 10-18 8 7" fill="none" stroke="#0b63ce" stroke-linecap="round" stroke-linejoin="round" stroke-width="5"></path><circle cx="43" cy="20" fill="#f5c84b" r="4"></circle></svg><span>Time-Series</span></span>
<span class="pf-badge"><svg aria-hidden="true" viewbox="0 0 64 64"><rect fill="#eaf4ff" height="27" rx="6" stroke="#0b63ce" stroke-width="3" width="36" x="14" y="28"></rect><path d="M22 28v-7c0-7 4-12 10-12s10 5 10 12v7" fill="none" stroke="#0b63ce" stroke-width="4"></path><circle cx="32" cy="41" fill="#f5c84b" r="4"></circle><path d="M32 45v5" stroke="#a36b00" stroke-linecap="round" stroke-width="3"></path></svg><span>Verified Lineage</span></span>
</div>
<div class="pf-stats">
<div class="pf-stat"><div class="pf-stat-value">26</div><div class="pf-stat-label">Phases · 34–59</div></div>
<div class="pf-stat"><div class="pf-stat-value">99–150</div><div class="pf-stat-label">Notebook Cell Range</div></div>
<div class="pf-stat"><div class="pf-stat-value">34–41</div><div class="pf-stat-label">V1 Recovered</div></div>
<div class="pf-stat"><div class="pf-stat-value">47–51 · 58–59</div><div class="pf-stat-label">V2 Final</div></div>
</div>
<div class="pf-flow">
<div class="pf-flow-node"><span>Recovered Sweeps</span><small>34–41 · V1</small></div><div class="pf-flow-arrow">→</div>
<div class="pf-flow-node"><span>Historical Pipeline</span><small>42–46 · V1</small></div><div class="pf-flow-arrow">→</div>
<div class="pf-flow-node"><span>Final Reporting</span><small>47–51 · V2</small></div><div class="pf-flow-arrow">→</div>
<div class="pf-flow-node"><span>Historical Attention</span><small>52–57 · V1</small></div><div class="pf-flow-arrow">→</div>
<div class="pf-flow-node"><span>Final Closure</span><small>58–59 · V2</small></div>
</div>
</div>
<div class="pf-document"><h1 class="pf-heading pf-h1">PHASE 34 TO 59 — Current Flow Detail</h1>

<blockquote class="pf-note">
<p>Mapping cell-level đầy đủ: <a class="pf-link" href="../link&amp;discussion_to_result/NOTEBOOK_CELLS_WALKTHROUGH.md">NOTEBOOK_CELLS_WALKTHROUGH.md</a>.</p>
</blockquote>
<div class="pf-nav-notice">
<div aria-hidden="true" class="pf-nav-notice-icon">↗</div>
<div>
<strong>Notebook navigation note</strong>
<p><code>Cells x–y</code> is the verified target range. <strong>Open Notebook ↗</strong> only opens <code>CourseWork.ipynb</code>; VS Code/Jupyter relative links do not reliably deep-link to an exact cell. Use the Phase navigation in this document for precise section jumps.</p>
</div>
</div><section class="pf-section pf-toc-card"><h2 class="pf-heading pf-h2 pf-toc-title">Mục lục</h2>
<ul class="pf-toc-list">
<li><a class="pf-link" href="#lineage-hiện-tại">Lineage hiện tại</a></li>
<li><a class="pf-link" href="#trạng-thái-recovery-hiện-tại-của-phase-3441">Trạng thái recovery Phase 34–41</a></li>
<li><strong>V1 recovered sweeps</strong><ul>
<li><a class="pf-link" href="#phase-34---s12-head-sweep">Phase 34 — S12 Head Sweep</a></li>
<li><a class="pf-link" href="#phase-35---s13-layer-sweep">Phase 35 — S13 Layer Sweep</a></li>
<li><a class="pf-link" href="#phase-36---s14-ffn-sweep">Phase 36 — S14 FFN Sweep</a></li>
<li><a class="pf-link" href="#phase-37---s15-loss-sweep">Phase 37 — S15 Loss Sweep</a></li>
<li><a class="pf-link" href="#phase-38---s16-epoch-cap-sweep">Phase 38 — S16 Epoch-Cap Sweep</a></li>
<li><a class="pf-link" href="#phase-39---s17-gradient-clipping-sweep">Phase 39 — S17 Gradient-Clipping Sweep</a></li>
<li><a class="pf-link" href="#phase-40---s18-revin-sweep">Phase 40 — S18 RevIN Sweep</a></li>
<li><a class="pf-link" href="#phase-41---s19-boundary-protocol-check">Phase 41 — S19 Boundary-Protocol Check</a></li>
</ul>
</li>
<li><strong>V1 historical final pipeline</strong><ul>
<li><a class="pf-link" href="#phase-42---candidate-synthesis">Phase 42 — Candidate Synthesis</a></li>
<li><a class="pf-link" href="#phase-43-lstm-tuning-results">Phase 43 — LSTM Tuning Results</a></li>
<li><a class="pf-link" href="#phase-44-rolling-origin-robustness-results">Phase 44 — Rolling-Origin Robustness Results</a></li>
<li><a class="pf-link" href="#phase-45-final-model-lock-results">Phase 45 — Final Model Lock Results</a></li>
<li><a class="pf-link" href="#phase-46-three-seed-final-run-results">Phase 46 — Three-Seed Final Run Results</a></li>
</ul>
</li>
<li><strong>V2 final reporting</strong><ul>
<li><a class="pf-link" href="#phase-47-final-v2-post-hoc-benchmark-results">Phase 47 — Final V2 Post-hoc Benchmark Results</a></li>
<li><a class="pf-link" href="#phase-48-final-v2-prediction-analysis-results">Phase 48 — Final V2 Prediction Analysis Results</a></li>
<li><a class="pf-link" href="#phase-49-final-v2-residual-analysis-results">Phase 49 — Final V2 Residual Analysis Results</a></li>
<li><a class="pf-link" href="#phase-50-final-v2-error-by-regime-analysis-results">Phase 50 — Final V2 Error-by-Regime Analysis Results</a></li>
<li><a class="pf-link" href="#phase-51-final-v2-worst-error-analysis-results">Phase 51 — Final V2 Worst-Error Analysis Results</a></li>
</ul>
</li>
<li><strong>V1 historical attention</strong><ul>
<li><a class="pf-link" href="#phase-52-historical-v1-attention-extraction-results">Phase 52 — Historical V1 Attention Extraction Results</a></li>
<li><a class="pf-link" href="#phase-53-historical-v1-attention-heatmap-results">Phase 53 — Historical V1 Attention Heatmap Results</a></li>
<li><a class="pf-link" href="#phase-54-historical-v1-last-query-attention-results">Phase 54 — Historical V1 Last-Query Attention Results</a></li>
<li><a class="pf-link" href="#phase-55-historical-v1-head-comparison-results">Phase 55 — Historical V1 Head Comparison Results</a></li>
<li><a class="pf-link" href="#phase-56-historical-v1-error-conditioned-attention-results">Phase 56 — Historical V1 Error-Conditioned Attention Results</a></li>
<li><a class="pf-link" href="#phase-57-historical-v1-seed-stability-attention-results">Phase 57 — Historical V1 Seed-Stability Attention Results</a></li>
</ul>
</li>
<li><strong>V2 final closure</strong><ul>
<li><a class="pf-link" href="#phase-58-final-v2-results-summary">Phase 58 — Final V2 Results Summary</a></li>
<li><a class="pf-link" href="#phase-59-final-v2-conclusions">Phase 59 — Final V2 Conclusions</a></li>
</ul>
</li>
<li><a class="pf-link" href="#tổng-kết-phases-34-59">Tổng kết Phases 34–59</a></li>
<li><a class="pf-link" href="#cấu-trúc-phân-tầng-sau-refactor">Cấu trúc phân tầng sau refactor</a></li>
</ul>
</section><section class="pf-section pf-lineage-card"><h2 class="pf-heading pf-h2" id="lineage-hiện-tại">Lineage hiện tại</h2>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Phase</th>
<th>Lineage</th>
<th>Quy tắc</th>
</tr>
</thead>
<tbody>
<tr>
<td>34–41</td>
<td><code class="pf-inline-code pf-code-blue">V1 RECOVERED</code></td>
<td><code class="pf-inline-code pf-code-blue">VALID_REUSABLE</code>, <code class="pf-inline-code pf-code-blue">RENDER_ONLY</code>; reporting debt không được fabricate.</td>
</tr>
<tr>
<td>42–46</td>
<td><code class="pf-inline-code pf-code-yellow">V1 HISTORICAL</code></td>
<td>Giữ historical final-pipeline provenance.</td>
</tr>
<tr>
<td>47–51</td>
<td><code class="pf-inline-code pf-code-green">V2 FINAL</code></td>
<td>Final locked V2 policy và reporting từ frozen Step 17 predictions.</td>
</tr>
<tr>
<td>52–57</td>
<td><code class="pf-inline-code pf-code-yellow">V1 HISTORICAL ATTENTION</code></td>
<td>Không phải attention evidence của final V2.</td>
</tr>
<tr>
<td>58–59</td>
<td><code class="pf-inline-code pf-code-green">V2 FINAL</code></td>
<td>V2 summary và conclusions.</td>
</tr>
</tbody>
</table></div>
</section><section class="pf-section pf-recovery-card"><h2 class="pf-heading pf-h2" id="trạng-thái-recovery-hiện-tại-của-phase-3441">Trạng thái recovery hiện tại của Phase 34–41</h2>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th style="text-align:right">Phase</th>
<th>State</th>
<th>Effective action</th>
<th>Scientific core</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right">34</td>
<td><code class="pf-inline-code pf-code-blue">VALID_REUSABLE</code></td>
<td><code class="pf-inline-code pf-code-blue">RENDER_ONLY</code></td>
<td>Verified; fail-closed compatibility</td>
</tr>
<tr>
<td style="text-align:right">35</td>
<td><code class="pf-inline-code pf-code-blue">VALID_REUSABLE</code></td>
<td><code class="pf-inline-code pf-code-blue">RENDER_ONLY</code></td>
<td>Verified; fail-closed compatibility</td>
</tr>
<tr>
<td style="text-align:right">36</td>
<td><code class="pf-inline-code pf-code-blue">VALID_REUSABLE</code></td>
<td><code class="pf-inline-code pf-code-blue">RENDER_ONLY</code></td>
<td>Verified; fail-closed compatibility</td>
</tr>
<tr>
<td style="text-align:right">37</td>
<td><code class="pf-inline-code pf-code-blue">VALID_REUSABLE</code></td>
<td><code class="pf-inline-code pf-code-blue">RENDER_ONLY</code></td>
<td>Verified; fail-closed compatibility</td>
</tr>
<tr>
<td style="text-align:right">38</td>
<td><code class="pf-inline-code pf-code-blue">VALID_REUSABLE</code></td>
<td><code class="pf-inline-code pf-code-blue">RENDER_ONLY</code></td>
<td>Verified; fail-closed compatibility</td>
</tr>
<tr>
<td style="text-align:right">39</td>
<td><code class="pf-inline-code pf-code-blue">VALID_REUSABLE</code></td>
<td><code class="pf-inline-code pf-code-blue">RENDER_ONLY</code></td>
<td>Verified; fail-closed compatibility</td>
</tr>
<tr>
<td style="text-align:right">40</td>
<td><code class="pf-inline-code pf-code-blue">VALID_REUSABLE</code></td>
<td><code class="pf-inline-code pf-code-blue">RENDER_ONLY</code></td>
<td>Verified; fail-closed compatibility</td>
</tr>
<tr>
<td style="text-align:right">41</td>
<td><code class="pf-inline-code pf-code-blue">VALID_REUSABLE</code></td>
<td><code class="pf-inline-code pf-code-blue">RENDER_ONLY</code></td>
<td>Verified; fail-closed compatibility</td>
</tr>
</tbody>
</table></div>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-recovered" data-phase="34"><h2 class="pf-heading pf-h2" id="phase-34---s12-head-sweep"><span aria-hidden="true" class="pf-phase-pill">PHASE 34</span>Phase 34 - S12 Head Sweep</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V1 RECOVERED</span><span class="pf-meta-chip">Cells 99–100</span><span class="pf-meta-chip pf-meta-state">RENDER_ONLY</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 99–100 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">phase-34-heading</code> → <code class="pf-inline-code">phase-34-resume</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/frozen_evidence.py">renderer</a> — <code class="pf-inline-code">render_frozen_phase_evidence(34, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/sweeps/heads.py">heads.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/sweeps/S12_heads/phase_34_signoff.json">phase_34_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-blue">V1 RECOVERED</code>.</li>
</ul>
<p><strong>Current state:</strong> <code class="pf-inline-code pf-code-blue">VALID_REUSABLE</code>; effective action <code class="pf-inline-code pf-code-blue">RENDER_ONLY</code>. Scientific core được fail-closed verify; historical reporting debt không được fabricate.</p>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/sweeps/heads.py</code></p>
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
<td>Phase 33 winner (d_model)</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S12_heads/"><code class="pf-inline-code">artifacts/sweeps/S12_heads/</code></a><code class="pf-inline-code">{sweep_manifest.json, results.csv, s12_head_winner.json, s12_reference_update.json}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S12_heads/phase_34_signoff.json"><code class="pf-inline-code">phase_34_signoff.json</code></a></td>
</tr>
<tr>
<td><strong>Output (figures)</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S12_heads/figures/"><code class="pf-inline-code">artifacts/sweeps/S12_heads/figures/</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Sweep <code class="pf-inline-code">num_heads ∈ {2, 4, 8, 16}</code></li>
<li>Validation RMSE, MAE, learning curves</li>
<li>Reference update</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-recovered" data-phase="35"><h2 class="pf-heading pf-h2" id="phase-35---s13-layer-sweep"><span aria-hidden="true" class="pf-phase-pill">PHASE 35</span>Phase 35 - S13 Layer Sweep</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V1 RECOVERED</span><span class="pf-meta-chip">Cells 101–102</span><span class="pf-meta-chip pf-meta-state">RENDER_ONLY</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 101–102 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">phase-35-heading</code> → <code class="pf-inline-code">phase-35-resume</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/frozen_evidence.py">renderer</a> — <code class="pf-inline-code">render_frozen_phase_evidence(35, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/sweeps/layers.py">layers.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/sweeps/S13_layers/phase_35_signoff.json">phase_35_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-blue">V1 RECOVERED</code>.</li>
</ul>
<p><strong>Current state:</strong> <code class="pf-inline-code pf-code-blue">VALID_REUSABLE</code>; effective action <code class="pf-inline-code pf-code-blue">RENDER_ONLY</code>. Scientific core được fail-closed verify; historical reporting debt không được fabricate.</p>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/sweeps/layers.py</code></p>
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
<td><a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S13_layers/"><code class="pf-inline-code">artifacts/sweeps/S13_layers/</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S13_layers/phase_35_signoff.json"><code class="pf-inline-code">phase_35_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Sweep <code class="pf-inline-code">num_layers ∈ {1, 2, 3, 4, 6}</code></li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-recovered" data-phase="36"><h2 class="pf-heading pf-h2" id="phase-36---s14-ffn-sweep"><span aria-hidden="true" class="pf-phase-pill">PHASE 36</span>Phase 36 - S14 FFN Sweep</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V1 RECOVERED</span><span class="pf-meta-chip">Cells 103–104</span><span class="pf-meta-chip pf-meta-state">RENDER_ONLY</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 103–104 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">phase-36-heading</code> → <code class="pf-inline-code">phase-36-resume</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/frozen_evidence.py">renderer</a> — <code class="pf-inline-code">render_frozen_phase_evidence(36, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/sweeps/ffn.py">ffn.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/sweeps/S14_ffn/phase_36_signoff.json">phase_36_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-blue">V1 RECOVERED</code>.</li>
</ul>
<p><strong>Current state:</strong> <code class="pf-inline-code pf-code-blue">VALID_REUSABLE</code>; effective action <code class="pf-inline-code pf-code-blue">RENDER_ONLY</code>. Scientific core được fail-closed verify; historical reporting debt không được fabricate.</p>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/sweeps/ffn.py</code></p>
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
<td><a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S14_ffn/"><code class="pf-inline-code">artifacts/sweeps/S14_ffn/</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S14_ffn/s14_ffn_metrics.csv"><code class="pf-inline-code">s14_ffn_metrics.csv</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S14_ffn/s14_ffn_winner.json"><code class="pf-inline-code">s14_ffn_winner.json</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S14_ffn/s14_reference_update.json"><code class="pf-inline-code">s14_reference_update.json</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S14_ffn/s14_ffn_sweep_manifest.json"><code class="pf-inline-code">s14_ffn_sweep_manifest.json</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S14_ffn/s14_ffn_sweep_summary.json"><code class="pf-inline-code">s14_ffn_sweep_summary.json</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S14_ffn/s14_ffn_sweep_report.md"><code class="pf-inline-code">s14_ffn_sweep_report.md</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S14_ffn/phase_36_signoff.json"><code class="pf-inline-code">phase_36_signoff.json</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S14_ffn/figures/"><code class="pf-inline-code">figures/</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Sweep FFN width multipliers</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-recovered" data-phase="37"><h2 class="pf-heading pf-h2" id="phase-37---s15-loss-sweep"><span aria-hidden="true" class="pf-phase-pill">PHASE 37</span>Phase 37 - S15 Loss Sweep</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V1 RECOVERED</span><span class="pf-meta-chip">Cells 105–106</span><span class="pf-meta-chip pf-meta-state">RENDER_ONLY</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 105–106 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">phase-37-heading</code> → <code class="pf-inline-code">phase-37-resume</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/frozen_evidence.py">renderer</a> — <code class="pf-inline-code">render_frozen_phase_evidence(37, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/sweeps/loss.py">loss.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/sweeps/S15_loss/phase_37_signoff.json">phase_37_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-blue">V1 RECOVERED</code>.</li>
</ul>
<p><strong>Current state:</strong> <code class="pf-inline-code pf-code-blue">VALID_REUSABLE</code>; effective action <code class="pf-inline-code pf-code-blue">RENDER_ONLY</code>. Scientific core được fail-closed verify; historical reporting debt không được fabricate.</p>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/sweeps/loss.py</code></p>
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
<td><a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S15_loss/"><code class="pf-inline-code">artifacts/sweeps/S15_loss/</code></a><code class="pf-inline-code">{s15_loss_winner.json, s15_reference_update.json}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S15_loss/phase_37_signoff.json"><code class="pf-inline-code">phase_37_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Sweep loss functions (MSE, Huber, Smooth L1)</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-recovered" data-phase="38"><h2 class="pf-heading pf-h2" id="phase-38---s16-epoch-cap-sweep"><span aria-hidden="true" class="pf-phase-pill">PHASE 38</span>Phase 38 - S16 Epoch-Cap Sweep</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V1 RECOVERED</span><span class="pf-meta-chip">Cells 107–108</span><span class="pf-meta-chip pf-meta-state">RENDER_ONLY</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 107–108 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">b3d754ee</code> → <code class="pf-inline-code">17dc5a66</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/frozen_evidence.py">renderer</a> — <code class="pf-inline-code">render_frozen_phase_evidence(38, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/sweeps/epoch_cap.py">epoch_cap.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/sweeps/S16_epoch_cap/phase_38_signoff.json">phase_38_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-blue">V1 RECOVERED</code>.</li>
</ul>
<p><strong>Current state:</strong> <code class="pf-inline-code pf-code-blue">VALID_REUSABLE</code>; effective action <code class="pf-inline-code pf-code-blue">RENDER_ONLY</code>. Scientific core được fail-closed verify; historical reporting debt không được fabricate.</p>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/sweeps/epoch_cap.py</code></p>
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
<td><a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S16_epoch_cap/"><code class="pf-inline-code">artifacts/sweeps/S16_epoch_cap/</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S16_epoch_cap/phase_38_signoff.json"><code class="pf-inline-code">phase_38_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Sweep max epochs cap</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-recovered" data-phase="39"><h2 class="pf-heading pf-h2" id="phase-39---s17-gradient-clipping-sweep"><span aria-hidden="true" class="pf-phase-pill">PHASE 39</span>Phase 39 - S17 Gradient-Clipping Sweep</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V1 RECOVERED</span><span class="pf-meta-chip">Cells 109–110</span><span class="pf-meta-chip pf-meta-state">RENDER_ONLY</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 109–110 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">d0833283</code> → <code class="pf-inline-code">f0fcdce7</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/frozen_evidence.py">renderer</a> — <code class="pf-inline-code">render_frozen_phase_evidence(39, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/sweeps/gradient_clip.py">gradient_clip.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/sweeps/S17_gradient_clipping/phase_39_signoff.json">phase_39_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-blue">V1 RECOVERED</code>.</li>
</ul>
<p><strong>Current state:</strong> <code class="pf-inline-code pf-code-blue">VALID_REUSABLE</code>; effective action <code class="pf-inline-code pf-code-blue">RENDER_ONLY</code>. Scientific core được fail-closed verify; historical reporting debt không được fabricate.</p>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/sweeps/gradient_clip.py</code></p>
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
<td><a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S17_gradient_clipping/"><code class="pf-inline-code">artifacts/sweeps/S17_gradient_clipping/</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S17_gradient_clipping/phase_39_signoff.json"><code class="pf-inline-code">phase_39_signoff.json</code></a></td>
</tr>
<tr>
<td><strong>Scripts</strong></td>
<td><code class="pf-inline-code">src/course_work/scripts/p39_*.py</code> (4 scripts)</td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Sweep gradient clipping thresholds</li>
<li>Strict best (gc0) verification</li>
<li>Compliance corrective</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-recovered" data-phase="40"><h2 class="pf-heading pf-h2" id="phase-40---s18-revin-sweep"><span aria-hidden="true" class="pf-phase-pill">PHASE 40</span>Phase 40 - S18 RevIN Sweep</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V1 RECOVERED</span><span class="pf-meta-chip">Cells 111–112</span><span class="pf-meta-chip pf-meta-state">RENDER_ONLY</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 111–112 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">8a7873a9</code> → <code class="pf-inline-code">d407f95c</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/frozen_evidence.py">renderer</a> — <code class="pf-inline-code">render_frozen_phase_evidence(40, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/sweeps/revin.py">revin.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/sweeps/S18_revin/phase_40_signoff.json">phase_40_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-blue">V1 RECOVERED</code>.</li>
</ul>
<p><strong>Current state:</strong> <code class="pf-inline-code pf-code-blue">VALID_REUSABLE</code>; effective action <code class="pf-inline-code pf-code-blue">RENDER_ONLY</code>. Scientific core được fail-closed verify; historical reporting debt không được fabricate.</p>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/sweeps/revin.py</code> + <code class="pf-inline-code">src/course_work/models/revin.py</code></p>
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
<td><a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S18_revin/"><code class="pf-inline-code">artifacts/sweeps/S18_revin/</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S18_revin/phase_40_signoff.json"><code class="pf-inline-code">phase_40_signoff.json</code></a></td>
</tr>
<tr>
<td><strong>Scripts</strong></td>
<td><code class="pf-inline-code">src/course_work/scripts/p40_*.py</code> (3 scripts)</td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Sweep RevIN applicability (on/off)</li>
<li>Strict best (rn1) verification</li>
<li>Scaler-bridge audit</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-recovered" data-phase="41"><h2 class="pf-heading pf-h2" id="phase-41---s19-boundary-protocol-check"><span aria-hidden="true" class="pf-phase-pill">PHASE 41</span>Phase 41 - S19 Boundary-Protocol Check</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V1 RECOVERED</span><span class="pf-meta-chip">Cells 113–114</span><span class="pf-meta-chip pf-meta-state">RENDER_ONLY</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 113–114 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">5acc48a2</code> → <code class="pf-inline-code">7e86663d</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/frozen_evidence.py">renderer</a> — <code class="pf-inline-code">render_frozen_phase_evidence(41, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/sweeps/boundary_protocol.py">boundary_protocol.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/sweeps/S19_boundary_protocol/phase_41_signoff.json">phase_41_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-blue">V1 RECOVERED</code>.</li>
</ul>
<p><strong>Current state:</strong> <code class="pf-inline-code pf-code-blue">VALID_REUSABLE</code>; effective action <code class="pf-inline-code pf-code-blue">RENDER_ONLY</code>. Scientific core được fail-closed verify; historical reporting debt không được fabricate.</p>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/sweeps/boundary_protocol.py</code></p>
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
<td><a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S19_boundary_protocol/"><code class="pf-inline-code">artifacts/sweeps/S19_boundary_protocol/</code></a>, <a class="pf-link pf-artifact-link" href="../../artifacts/sweeps/S19_boundary_protocol/phase_41_signoff.json"><code class="pf-inline-code">phase_41_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Test boundary handling for windowing</li>
<li>Validate no leakage at boundaries</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-historical" data-phase="42"><h2 class="pf-heading pf-h2" id="phase-42---candidate-synthesis"><span aria-hidden="true" class="pf-phase-pill">PHASE 42</span>Phase 42 - Candidate Synthesis</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V1 HISTORICAL</span><span class="pf-meta-chip">Cells 115–116</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 115–116 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">fa11ad6a</code> → <code class="pf-inline-code">994bbdb9</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/phase_summary.py">renderer</a> — <code class="pf-inline-code">render_phase_summary(42, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/scripts/p42_candidate_synthesis.py">p42_candidate_synthesis.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/candidate_synthesis/phase_42_signoff.json">phase_42_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-yellow">V1 HISTORICAL</code>.</li>
</ul>
<p><strong>Current scope:</strong> <code class="pf-inline-code pf-code-yellow">V1 HISTORICAL</code>. Nội dung bên dưới mô tả scientific workflow lịch sử; notebook hiện chỉ render verified artifacts.</p>
<p><strong>Script:</strong> <code class="pf-inline-code">src/course_work/scripts/p42_candidate_synthesis.py</code></p>
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
<td>Phase 23-41 sweep winners</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/candidate_synthesis/"><code class="pf-inline-code">artifacts/candidate_synthesis/</code></a><code class="pf-inline-code">{candidate_synthesis_manifest.json, candidate_synthesis_report.md, transformer_candidate_shortlist.json, selected_lineage.json, baseline_anchor_context.json, boundary_sensitivity_context.json}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/candidate_synthesis/phase_42_signoff.json"><code class="pf-inline-code">phase_42_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Tổng hợp các sweep winners</li>
<li>Tạo candidate shortlist cho Transformer</li>
<li>Phase handoff to LSTM tuning</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-historical" data-phase="43"><h2 class="pf-heading pf-h2" id="phase-43-lstm-tuning-results"><span aria-hidden="true" class="pf-phase-pill">PHASE 43</span>Phase 43 — LSTM Tuning Results</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V1 HISTORICAL</span><span class="pf-meta-chip">Cells 117–118</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 117–118 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">ed1e48a7</code> → <code class="pf-inline-code">6a082d01</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/results_rebuild.py">renderer</a> — <code class="pf-inline-code">render_verified_phase_result(43, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/scripts/p43_lstm_tuning.py">p43_lstm_tuning.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/lstm_tuning/phase_43_signoff.json">phase_43_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-yellow">V1 HISTORICAL</code>.</li>
</ul>
<p><strong>Current scope:</strong> <code class="pf-inline-code pf-code-yellow">V1 HISTORICAL</code>. Nội dung bên dưới mô tả scientific workflow lịch sử; notebook hiện chỉ render verified artifacts.</p>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/lstm_tuning/{stages, tuning_space, winners, ...}.py</code></p>
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
<td><a class="pf-link pf-artifact-link" href="../../artifacts/lstm_tuning/"><code class="pf-inline-code">artifacts/lstm_tuning/</code></a><code class="pf-inline-code">{lstm_tuning_manifest.json, lstm_tuning_report.md, lstm_tuned_winner.json, lt{1,2,3,4,5}_*_winner.json}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/lstm_tuning/phase_43_signoff.json"><code class="pf-inline-code">phase_43_signoff.json</code></a></td>
</tr>
<tr>
<td><strong>Output (figures)</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/lstm_tuning/figures/"><code class="pf-inline-code">artifacts/lstm_tuning/figures/</code></a></td>
</tr>
<tr>
<td><strong>Scripts</strong></td>
<td><code class="pf-inline-code">src/course_work/scripts/p43_*.py</code> (6 scripts)</td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Sweep LSTM hyperparameters (hidden_size, layers, dropout, lr, weight_decay)</li>
<li>Sequential stages (lt1-lt5)</li>
<li>Output LSTM tuned winner</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-historical" data-phase="44"><h2 class="pf-heading pf-h2" id="phase-44-rolling-origin-robustness-results"><span aria-hidden="true" class="pf-phase-pill">PHASE 44</span>Phase 44 — Rolling-Origin Robustness Results</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V1 HISTORICAL</span><span class="pf-meta-chip">Cells 119–120</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 119–120 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">9eacc9e3</code> → <code class="pf-inline-code">2f0130ea</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/results_rebuild.py">renderer</a> — <code class="pf-inline-code">render_verified_phase_result(44, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/scripts/p44_rolling_origin.py">p44_rolling_origin.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/rolling_origin/phase_44_signoff.json">phase_44_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-yellow">V1 HISTORICAL</code>.</li>
</ul>
<p><strong>Current scope:</strong> <code class="pf-inline-code pf-code-yellow">V1 HISTORICAL</code>. Nội dung bên dưới mô tả scientific workflow lịch sử; notebook hiện chỉ render verified artifacts.</p>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/rolling_origin/{folds, persistence, refit_engine, ...}.py</code></p>
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
<td>Phase 42 candidates + Phase 43 LSTM tuned</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/rolling_origin/"><code class="pf-inline-code">artifacts/rolling_origin/</code></a><code class="pf-inline-code">{rolling_origin_manifest.json, rolling_origin_summary.json, rolling_origin_report.md, rolling_origin_fold_manifest.json, rolling_origin_fold_local_scaling_contract.json, rolling_origin_recommended_transformer.json}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/rolling_origin/phase_44_signoff.json"><code class="pf-inline-code">phase_44_signoff.json</code></a></td>
</tr>
<tr>
<td><strong>Output (figures)</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/rolling_origin/figures/"><code class="pf-inline-code">artifacts/rolling_origin/figures/</code></a></td>
</tr>
<tr>
<td><strong>Scripts</strong></td>
<td><code class="pf-inline-code">src/course_work/scripts/p44_*.py</code> (3 scripts)</td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Rolling-origin cross-validation</li>
<li>Test generalization across time slices</li>
<li>Robust Lane (full refit on each fold) vs Fast Lane (warm-start)</li>
<li>Pick recommended Transformer candidate</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-historical" data-phase="45"><h2 class="pf-heading pf-h2" id="phase-45-final-model-lock-results"><span aria-hidden="true" class="pf-phase-pill">PHASE 45</span>Phase 45 — Final Model Lock Results</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V1 HISTORICAL</span><span class="pf-meta-chip">Cells 121–122</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 121–122 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">88d07113</code> → <code class="pf-inline-code">4e6b273d</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/results_rebuild.py">renderer</a> — <code class="pf-inline-code">render_verified_phase_result(45, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/scripts/p45_final_model_lock.py">p45_final_model_lock.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/final_model_lock/final_model_lock_summary.json">final_model_lock_summary.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-yellow">V1 HISTORICAL</code>.</li>
</ul>
<p><strong>Current scope:</strong> <code class="pf-inline-code pf-code-yellow">V1 HISTORICAL</code>. Nội dung bên dưới mô tả scientific workflow lịch sử; notebook hiện chỉ render verified artifacts.</p>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/final_model_lock/{candidate_lock, recipe, lineage, fingerprints, ...}.py</code></p>
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
<td>Phase 44 rolling origin results</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/final_model_lock/"><code class="pf-inline-code">artifacts/final_model_lock/</code></a><code class="pf-inline-code">{final_model_lock_manifest.json, final_model_lock_contract.json, final_model_lock_fingerprint.json, final_model_lock_summary.json, final_*_contract.json, final_*_fingerprint.json, final_*_evidence.json}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/final_model_lock/phase_45_signoff.json"><code class="pf-inline-code">phase_45_signoff.json</code></a></td>
</tr>
<tr>
<td><strong>Scripts</strong></td>
<td><code class="pf-inline-code">src/course_work/scripts/p45_*.py</code> (2 scripts)</td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li><strong>NO-TRAIN phase</strong> — chỉ lock config, KHÔNG train</li>
<li>Freeze final model config: architecture, optimizer, loss, scaler, RevIN, seed, feature set, data region, environment</li>
<li>Generate fingerprints for every contract</li>
<li>Phase handoff to three-seed runs</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-historical" data-phase="46"><h2 class="pf-heading pf-h2" id="phase-46-three-seed-final-run-results"><span aria-hidden="true" class="pf-phase-pill">PHASE 46</span>Phase 46 — Three-Seed Final Run Results</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V1 HISTORICAL</span><span class="pf-meta-chip">Cells 123–124</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 123–124 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">b388dcd1</code> → <code class="pf-inline-code">753ab0f1</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/results_rebuild.py">renderer</a> — <code class="pf-inline-code">render_verified_phase_result(46, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/scripts/p46_three_seed_runs.py">p46_three_seed_runs.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/three_seed_final_runs/phase_46_signoff.json">phase_46_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-yellow">V1 HISTORICAL</code>.</li>
</ul>
<p><strong>Current scope:</strong> <code class="pf-inline-code pf-code-yellow">V1 HISTORICAL</code>. Nội dung bên dưới mô tả scientific workflow lịch sử; notebook hiện chỉ render verified artifacts.</p>
<p><strong>Script:</strong> <code class="pf-inline-code">src/course_work/scripts/p46_three_seed_runs.py</code></p>
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
<td>Phase 45 locked config</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/three_seed_final_runs/"><code class="pf-inline-code">artifacts/three_seed_final_runs/</code></a><code class="pf-inline-code">{three_seed_manifest.json, three_seed_contract.json, three_seed_final_runs_summary.json, final_lock_verification.json, final_dev_population_manifest.json, phase47_final_test_evaluation_handoff.json}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/three_seed_final_runs/phase_46_signoff.json"><code class="pf-inline-code">phase_46_signoff.json</code></a></td>
</tr>
<tr>
<td><strong>Output (checkpoints)</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/three_seed_final_runs/official_checkpoints/"><code class="pf-inline-code">artifacts/three_seed_final_runs/official_checkpoints/</code></a><code class="pf-inline-code">seed_{42,123,2026}/*_FINAL_REFIT_metadata.json</code></td>
</tr>
<tr>
<td><strong>Output (figures)</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/three_seed_final_runs/figures/"><code class="pf-inline-code">artifacts/three_seed_final_runs/figures/</code></a></td>
</tr>
<tr>
<td><strong>Scripts</strong></td>
<td><code class="pf-inline-code">src/course_work/scripts/p46_*.py</code> (7 scripts: archive, preflight, gate, smoke, schema_sim, registration, three_seed)</td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Train final model với 3 seeds: 42, 123, 2026</li>
<li>Each seed produces a FINAL_REFIT checkpoint + metadata</li>
<li>Verify against Phase 45 lock</li>
<li>Phase handoff to final test evaluation</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-v2final" data-phase="47"><h2 class="pf-heading pf-h2" id="phase-47-final-v2-post-hoc-benchmark-results"><span aria-hidden="true" class="pf-phase-pill">PHASE 47</span>Phase 47 — Final V2 Post-hoc Benchmark Results</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V2 FINAL</span><span class="pf-meta-chip">Cells 125–126</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 125–126 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">62f722b6</code> → <code class="pf-inline-code">4dea0c14</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/results_rebuild.py">renderer</a> — <code class="pf-inline-code">render_verified_phase_result(47, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/model_improvement_v2/step17_mape_addendum.py">step17_mape_addendum.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics_with_mape.json">step17_metrics_with_mape.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-green">V2 FINAL</code>.</li>
</ul>
<p><strong>Current V2 presentation:</strong> Phase 47 đọc <code class="pf-inline-code">step17_metrics_with_mape.json</code> và trình bày <code class="pf-inline-code">POST_HOC_V2_BENCHMARK</code> cho ba seed, equal-weight ensemble và Persistence.</p>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Model / policy</th>
<th style="text-align:right">RMSE Wh</th>
<th style="text-align:right">MAE Wh</th>
<th style="text-align:right">MAPE %</th>
<th style="text-align:right">R²</th>
</tr>
</thead>
<tbody>
<tr>
<td>V2 Equal-weight Ensemble</td>
<td style="text-align:right">61.608937</td>
<td style="text-align:right">25.897130</td>
<td style="text-align:right">21.519219</td>
<td style="text-align:right">0.540364</td>
</tr>
<tr>
<td>Persistence</td>
<td style="text-align:right">66.836915</td>
<td style="text-align:right">26.737589</td>
<td style="text-align:right">21.513353</td>
<td style="text-align:right">0.459047</td>
</tr>
</tbody>
</table></div>
<p><strong>Historical V1 workflow context (provenance only):</strong></p>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/final_test_evaluation/{evaluation, checkpoint_loader, scaler_loader, ...}.py</code></p>
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
<td>Phase 46 official checkpoints</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/final_test/"><code class="pf-inline-code">artifacts/final_test/</code></a><code class="pf-inline-code">{final_test_evaluation_contract.json, final_test_evaluation_manifest.json, final_test_release_verification.json, final_test_population_manifest.json, final_test_lstm_eligibility.json, final_test_access_event.json, final_test_access_log.jsonl, final_test_discrepancies.json, final_test_summary.json, final_test_report.md, prediction_checksums.json}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/final_test/phase_47_signoff.json"><code class="pf-inline-code">phase_47_signoff.json</code></a></td>
</tr>
<tr>
<td><strong>Output (figures)</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/final_test/figures/"><code class="pf-inline-code">artifacts/final_test/figures/</code></a></td>
</tr>
<tr>
<td><strong>Handoffs</strong></td>
<td><code class="pf-inline-code">phase48_prediction_analysis_handoff.json</code>, <code class="pf-inline-code">phase49_residual_analysis_handoff.json</code>, <code class="pf-inline-code">phase50_error_regime_handoff.json</code>, <code class="pf-inline-code">phase51_worst_error_handoff.json</code>, <code class="pf-inline-code">phase52_attention_extraction_handoff.json</code></td>
</tr>
<tr>
<td><strong>Scripts</strong></td>
<td><code class="pf-inline-code">src/course_work/scripts/p47_*.py</code> (2 scripts)</td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li><strong>FINAL GATE</strong> — Đánh giá cuối cùng trên Test set</li>
<li>Sử dụng checkpoints đã trained từ Phase 46</li>
<li>Test access events audit (ghi lại mọi lần truy cập Test)</li>
<li>Generate final test report với metrics cho cả 3 seeds</li>
<li>Phase handoffs cho analysis phases 48-52</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-v2final" data-phase="48"><h2 class="pf-heading pf-h2" id="phase-48-final-v2-prediction-analysis-results"><span aria-hidden="true" class="pf-phase-pill">PHASE 48</span>Phase 48 — Final V2 Prediction Analysis Results</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V2 FINAL</span><span class="pf-meta-chip">Cells 127–128</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 127–128 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">b496ac6f</code> → <code class="pf-inline-code">1451ccf0</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/results_rebuild.py">renderer</a> — <code class="pf-inline-code">render_verified_phase_result(48, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/reporting/v2_final_analysis.py">v2_final_analysis.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/model_improvement_v2/final_reporting_analysis/phase48_prediction_analysis.json">phase48_prediction_analysis.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-green">V2 FINAL</code>.</li>
</ul>
<p><strong>Current V2 presentation:</strong> Reporting-only prediction distribution/change/peak/Persistence comparison từ frozen Step 17 predictions.</p>
<p><strong>Historical V1 workflow context (provenance only):</strong></p>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/analysis/prediction_analysis/{alignment, distribution, change_behavior, ...}.py</code></p>
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
<td>Phase 47 final test predictions</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/prediction_analysis/"><code class="pf-inline-code">artifacts/prediction_analysis/</code></a><code class="pf-inline-code">{prediction_analysis_manifest.json, prediction_analysis_summary.json, prediction_analysis_report.md}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/prediction_analysis/phase_48_signoff.json"><code class="pf-inline-code">phase_48_signoff.json</code></a></td>
</tr>
<tr>
<td><strong>Output (figures)</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/prediction_analysis/figures/"><code class="pf-inline-code">artifacts/prediction_analysis/figures/</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Actual vs Predicted (full test)</li>
<li>Zoom views (first/middle/last 24h)</li>
<li>Per-seed scatter, ECDF</li>
<li>Change magnitude distribution</li>
<li>Cross-seed spread over time</li>
<li>Lag cross-correlation</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-v2final" data-phase="49"><h2 class="pf-heading pf-h2" id="phase-49-final-v2-residual-analysis-results"><span aria-hidden="true" class="pf-phase-pill">PHASE 49</span>Phase 49 — Final V2 Residual Analysis Results</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V2 FINAL</span><span class="pf-meta-chip">Cells 129–130</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 129–130 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">7cc29214</code> → <code class="pf-inline-code">bbcf08f2</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/results_rebuild.py">renderer</a> — <code class="pf-inline-code">render_verified_phase_result(49, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/reporting/v2_final_analysis.py">v2_final_analysis.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/model_improvement_v2/final_reporting_analysis/phase49_residual_analysis.json">phase49_residual_analysis.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-green">V2 FINAL</code>.</li>
</ul>
<p><strong>Current V2 presentation:</strong> Residual/bias/error-distribution analysis của final equal-weight ensemble và Persistence.</p>
<p><strong>Historical V1 workflow context (provenance only):</strong></p>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/analysis/residual_analysis/{distributions, bias, autocorrelation, ...}.py</code></p>
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
<td>Phase 47 + 48</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/residual_analysis/"><code class="pf-inline-code">artifacts/residual_analysis/</code></a><code class="pf-inline-code">{phase49_*.json, phase49_report.json}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/residual_analysis/phase_49_signoff.json"><code class="pf-inline-code">phase_49_signoff.json</code></a></td>
</tr>
<tr>
<td><strong>Output (figures)</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/residual_analysis/figures/"><code class="pf-inline-code">artifacts/residual_analysis/figures/</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Residual time series, distribution, ECDF</li>
<li>Signed bias, sign balance</li>
<li>Tail diagnostics</li>
<li>ACF, sign runs, sign transitions</li>
<li>Rolling statistics</li>
<li>Magnitude associations (residual vs y_true, y_pred)</li>
<li>Cross-seed agreement</li>
<li>Persistence context</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-v2final" data-phase="50"><h2 class="pf-heading pf-h2" id="phase-50-final-v2-error-by-regime-analysis-results"><span aria-hidden="true" class="pf-phase-pill">PHASE 50</span>Phase 50 — Final V2 Error-by-Regime Analysis Results</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V2 FINAL</span><span class="pf-meta-chip">Cells 131–132</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 131–132 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">9a2f1b89</code> → <code class="pf-inline-code">1d5b5327</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/results_rebuild.py">renderer</a> — <code class="pf-inline-code">render_verified_phase_result(50, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/reporting/v2_final_analysis.py">v2_final_analysis.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/model_improvement_v2/final_reporting_analysis/phase50_error_by_regime.json">phase50_error_by_regime.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-green">V2 FINAL</code>.</li>
</ul>
<p><strong>Current V2 presentation:</strong> Error-by-regime dùng verified train-defined thresholds; <code class="pf-inline-code">population_equality=PASS</code>.</p>
<p><strong>Historical V1 workflow context (provenance only):</strong></p>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/analysis/error_regime_analysis/{regime_assignment, thresholds, cross_seed, ...}.py</code></p>
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
<td>Phase 47 + 49</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/error_by_regime/"><code class="pf-inline-code">artifacts/error_by_regime/</code></a><code class="pf-inline-code">{phase50_findings.json, phase50_report.md, regime_*_fingerprint.json, regime_*_mapping.json, regime_thresholds_train_only.json, test_regime_assignment_fingerprint.json}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/error_by_regime/phase_50_signoff.json"><code class="pf-inline-code">phase_50_signoff.json</code></a></td>
</tr>
<tr>
<td><strong>Output (figures)</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/error_by_regime/figures/"><code class="pf-inline-code">artifacts/error_by_regime/figures/</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Define regimes: target level, time of day, day type, extreme high, change direction, change magnitude</li>
<li>Compute regime thresholds (Train only)</li>
<li>Assign regimes to Test samples</li>
<li>Per-regime metrics (RMSE, MAE)</li>
<li>Cross-seed regime stability</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-v2final" data-phase="51"><h2 class="pf-heading pf-h2" id="phase-51-final-v2-worst-error-analysis-results"><span aria-hidden="true" class="pf-phase-pill">PHASE 51</span>Phase 51 — Final V2 Worst-Error Analysis Results</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V2 FINAL</span><span class="pf-meta-chip">Cells 133–134</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 133–134 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">2e8f083e</code> → <code class="pf-inline-code">c9df346a</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/results_rebuild.py">renderer</a> — <code class="pf-inline-code">render_verified_phase_result(51, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/reporting/v2_final_analysis.py">v2_final_analysis.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/model_improvement_v2/final_reporting_analysis/phase51_worst_error_analysis.json">phase51_worst_error_analysis.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-green">V2 FINAL</code>.</li>
</ul>
<p><strong>Current V2 presentation:</strong> Top-20 worst-error ranking/case summary của final ensemble từ frozen predictions.</p>
<p><strong>Historical V1 workflow context (provenance only):</strong></p>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/analysis/worst_error_analysis/{ranking, hardness, casebook, regime_context, ...}.py</code></p>
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
<td>Phase 47, 50</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/worst_error_analysis/"><code class="pf-inline-code">artifacts/worst_error_analysis/</code></a><code class="pf-inline-code">{worst_error_analysis_manifest.json, worst_error_ranking_manifest.json, worst_case_casebook.md, phase51_findings.json, phase51_report.md, phase51_summary.json, exact_input_reconstruction_manifest.json, lstm_eligibility_context.json, selection_contract_fingerprint.json}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/worst_error_analysis/phase51_signoff.json"><code class="pf-inline-code">phase51_signoff.json</code></a></td>
</tr>
<tr>
<td><strong>Output (figures)</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/worst_error_analysis/figures/"><code class="pf-inline-code">artifacts/worst_error_analysis/figures/</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Rank top-K worst errors per seed</li>
<li>Identify shared worst cases across seeds</li>
<li>Build casebook (markdown)</li>
<li>Regime context for worst cases</li>
<li>Exact input reconstruction (provenance)</li>
<li>LSTM eligibility context (chỉ dùng LSTM nếu được phép)</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-attention" data-phase="52"><h2 class="pf-heading pf-h2" id="phase-52-historical-v1-attention-extraction-results"><span aria-hidden="true" class="pf-phase-pill">PHASE 52</span>Phase 52 — Historical V1 Attention Extraction Results</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V1 HISTORICAL ATTENTION</span><span class="pf-meta-chip">Cells 135–136</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 135–136 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">870efbe4</code> → <code class="pf-inline-code">7b48186f</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/results_rebuild.py">renderer</a> — <code class="pf-inline-code">render_verified_phase_result(52, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/analysis/attention_extraction/orchestrator.py">orchestrator.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/attention_extraction/attention_extraction_summary.json">attention_extraction_summary.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-yellow">V1 HISTORICAL ATTENTION</code>.</li>
</ul>
<p><strong>Current scope:</strong> <code class="pf-inline-code pf-code-yellow">V1 HISTORICAL ATTENTION</code>. Không được diễn giải nội dung này như attention evidence của final V2.</p>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/analysis/attention_extraction/{extract, finalize, inputs, materialization, ...}.py</code></p>
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
<td>Phase 46 checkpoints + Phase 51 case IDs</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/attention_extraction/"><code class="pf-inline-code">artifacts/attention_extraction/</code></a><code class="pf-inline-code">{attention_extraction_manifest.json, attention_extraction_summary.json, attention_extraction_report.md, raw_attention_checksums.json, o52_inventory.json}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/attention_extraction/phase_52_signoff.json"><code class="pf-inline-code">phase_52_signoff.json</code></a></td>
</tr>
<tr>
<td><strong>Handoffs</strong></td>
<td><code class="pf-inline-code">phase53_attention_heatmaps_handoff.json</code>, <code class="pf-inline-code">phase54_last_query_attention_handoff.json</code>, ... (5 handoffs)</td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Extract attention weights for shared worst cases</li>
<li>Per-seed × per-layer × per-head</li>
<li>Checksums cho raw attention tensors</li>
<li>Phase handoffs cho attention analyses 53-57</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-attention" data-phase="53"><h2 class="pf-heading pf-h2" id="phase-53-historical-v1-attention-heatmap-results"><span aria-hidden="true" class="pf-phase-pill">PHASE 53</span>Phase 53 — Historical V1 Attention Heatmap Results</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V1 HISTORICAL ATTENTION</span><span class="pf-meta-chip">Cells 137–138</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 137–138 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">afeb48a2</code> → <code class="pf-inline-code">1c098fa6</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/results_rebuild.py">renderer</a> — <code class="pf-inline-code">render_verified_phase_result(53, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/analysis/attention_heatmaps/orchestrator.py">orchestrator.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/attention_heatmaps/phase_53_signoff.json">phase_53_signoff.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-yellow">V1 HISTORICAL ATTENTION</code>.</li>
</ul>
<p><strong>Current scope:</strong> <code class="pf-inline-code pf-code-yellow">V1 HISTORICAL ATTENTION</code>. Không được diễn giải nội dung này như attention evidence của final V2.</p>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/analysis/attention_heatmaps/{rendering, orientation, scales, catalog, ...}.py</code></p>
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
<td>Phase 52 raw attention</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/attention_heatmaps/"><code class="pf-inline-code">artifacts/attention_heatmaps/</code></a><code class="pf-inline-code">{attention_heatmaps_manifest.json, attention_heatmap_summary.json, attention_heatmap_report.md, attention_heatmap_catalog.md, attention_heatmap_render_config.json, attention_heatmap_image_checksums.json}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/attention_heatmaps/phase_53_signoff.json"><code class="pf-inline-code">phase_53_signoff.json</code></a></td>
</tr>
<tr>
<td><strong>Output (figures)</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/attention_heatmaps/"><code class="pf-inline-code">artifacts/attention_heatmaps/</code></a><code class="pf-inline-code">{case_grids, cross_seed, fixed_probability, individual_maps}/</code></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Render attention heatmaps (V1: case grids, V2: cross-seed, V3: individual maps)</li>
<li>Fixed probability mode + case-shared scale mode</li>
<li>Cross-orientation audit</li>
<li>Render config fingerprint</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-attention" data-phase="54"><h2 class="pf-heading pf-h2" id="phase-54-historical-v1-last-query-attention-results"><span aria-hidden="true" class="pf-phase-pill">PHASE 54</span>Phase 54 — Historical V1 Last-Query Attention Results</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V1 HISTORICAL ATTENTION</span><span class="pf-meta-chip">Cells 139–140</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 139–140 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">3c915fcc</code> → <code class="pf-inline-code">ec7b2ea5</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/results_rebuild.py">renderer</a> — <code class="pf-inline-code">render_verified_phase_result(54, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/analysis/last_query_attention/orchestrator.py">orchestrator.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/last_query_attention/last_query_attention_summary.json">last_query_attention_summary.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-yellow">V1 HISTORICAL ATTENTION</code>.</li>
</ul>
<p><strong>Current scope:</strong> <code class="pf-inline-code pf-code-yellow">V1 HISTORICAL ATTENTION</code>. Không được diễn giải nội dung này như attention evidence của final V2.</p>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/analysis/last_query_attention/{aggregations, profiles, top1_freq, coverage, ...}.py</code></p>
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
<td>Phase 52</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/last_query_attention/"><code class="pf-inline-code">artifacts/last_query_attention/</code></a><code class="pf-inline-code">{last_query_attention_manifest.json, last_query_attention_summary.json, last_query_attention_report.md, o54_inventory.json}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/last_query_attention/phase_54_signoff.json"><code class="pf-inline-code">phase_54_signoff.json</code></a></td>
</tr>
<tr>
<td><strong>Output (figures)</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/last_query_attention/figures/"><code class="pf-inline-code">artifacts/last_query_attention/figures/</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Focus on attention từ last query token</li>
<li>Top1 frequency, normalized entropy, expected lag</li>
<li>Lag-bin coverage, recency mass</li>
<li>Cumulative attention profiles</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-attention" data-phase="55"><h2 class="pf-heading pf-h2" id="phase-55-historical-v1-head-comparison-results"><span aria-hidden="true" class="pf-phase-pill">PHASE 55</span>Phase 55 — Historical V1 Head Comparison Results</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V1 HISTORICAL ATTENTION</span><span class="pf-meta-chip">Cells 141–142</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 141–142 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">644dbb41</code> → <code class="pf-inline-code">05ec07a2</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/results_rebuild.py">renderer</a> — <code class="pf-inline-code">render_verified_phase_result(55, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/analysis/head_comparison/orchestrator.py">orchestrator.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/head_comparison/head_comparison_summary.json">head_comparison_summary.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-yellow">V1 HISTORICAL ATTENTION</code>.</li>
</ul>
<p><strong>Current scope:</strong> <code class="pf-inline-code pf-code-yellow">V1 HISTORICAL ATTENTION</code>. Không được diễn giải nội dung này như attention evidence của final V2.</p>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/analysis/head_comparison/{matrices, behavior_cards, profile_metrics, ...}.py</code></p>
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
<td>Phase 52 + 54</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/head_comparison/"><code class="pf-inline-code">artifacts/head_comparison/</code></a><code class="pf-inline-code">{head_comparison_manifest.json, head_comparison_summary.json, head_comparison_report.md}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/head_comparison/phase_55_signoff.json"><code class="pf-inline-code">phase_55_signoff.json</code></a></td>
</tr>
<tr>
<td><strong>Output (figures)</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/head_comparison/figures/"><code class="pf-inline-code">artifacts/head_comparison/figures/</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Paired head metrics (JSD, cosine, Wasserstein)</li>
<li>Behavior cards cho mỗi head</li>
<li>Layer-head diversity summary</li>
<li>Mean temporal profiles by head</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-attention" data-phase="56"><h2 class="pf-heading pf-h2" id="phase-56-historical-v1-error-conditioned-attention-results"><span aria-hidden="true" class="pf-phase-pill">PHASE 56</span>Phase 56 — Historical V1 Error-Conditioned Attention Results</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V1 HISTORICAL ATTENTION</span><span class="pf-meta-chip">Cells 143–144</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 143–144 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">de13c9d0</code> → <code class="pf-inline-code">5e9eec01</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/results_rebuild.py">renderer</a> — <code class="pf-inline-code">render_verified_phase_result(56, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/analysis/error_conditioned_attention/orchestrator.py">orchestrator.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/error_conditioned_attention/error_conditioned_attention_summary.json">error_conditioned_attention_summary.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-yellow">V1 HISTORICAL ATTENTION</code>.</li>
</ul>
<p><strong>Current scope:</strong> <code class="pf-inline-code pf-code-yellow">V1 HISTORICAL ATTENTION</code>. Không được diễn giải nội dung này như attention evidence của final V2.</p>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/analysis/error_conditioned_attention/{analyses, cohort, core_metrics, ...}.py</code></p>
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
<td>Phase 51 case IDs + Phase 52 attention</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/error_conditioned_attention/"><code class="pf-inline-code">artifacts/error_conditioned_attention/</code></a><code class="pf-inline-code">{error_conditioned_attention_manifest.json, error_conditioned_attention_summary.json, error_conditioned_attention_report.md, error_conditioning_assignment_fingerprint.json, shared_error_conditioning_audit.json}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/error_conditioned_attention/phase_56_signoff.json"><code class="pf-inline-code">phase_56_signoff.json</code></a></td>
</tr>
<tr>
<td><strong>Output (figures)</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/error_conditioned_attention/figures/"><code class="pf-inline-code">artifacts/error_conditioned_attention/figures/</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>So sánh attention cho high-error vs low-error samples</li>
<li>Cliffs Delta, signed error</li>
<li>High vs low profile, decile trends</li>
<li>Shared cohort analysis</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-attention" data-phase="57"><h2 class="pf-heading pf-h2" id="phase-57-historical-v1-seed-stability-attention-results"><span aria-hidden="true" class="pf-phase-pill">PHASE 57</span>Phase 57 — Historical V1 Seed-Stability Attention Results</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V1 HISTORICAL ATTENTION</span><span class="pf-meta-chip">Cells 145–146</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 145–146 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">5627f983</code> → <code class="pf-inline-code">4293c1aa</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/results_rebuild.py">renderer</a> — <code class="pf-inline-code">render_verified_phase_result(57, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/analysis/seed_stability_attention/orchestrator.py">orchestrator.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/seed_stability_attention/seed_stability_attention_summary.json">seed_stability_attention_summary.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-yellow">V1 HISTORICAL ATTENTION</code>.</li>
</ul>
<p><strong>Current scope:</strong> <code class="pf-inline-code pf-code-yellow">V1 HISTORICAL ATTENTION</code>. Không được diễn giải nội dung này như attention evidence của final V2.</p>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/analysis/seed_stability_attention/{analyses, core_metrics, ...}.py</code></p>
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
<td>Phase 52</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/seed_stability_attention/"><code class="pf-inline-code">artifacts/seed_stability_attention/</code></a><code class="pf-inline-code">{seed_stability_attention_manifest.json, seed_stability_attention_summary.json, seed_stability_attention_report.md, head_matching_fingerprint.json}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/seed_stability_attention/phase_57_signoff.json"><code class="pf-inline-code">phase_57_signoff.json</code></a></td>
</tr>
<tr>
<td><strong>Output (figures)</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/seed_stability_attention/figures/"><code class="pf-inline-code">artifacts/seed_stability_attention/figures/</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Cross-seed attention stability</li>
<li>Canonical head mapping</li>
<li>Layer/head stability metrics</li>
<li>Prediction spread vs attention disagreement</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-v2closure" data-phase="58"><h2 class="pf-heading pf-h2" id="phase-58-final-v2-results-summary"><span aria-hidden="true" class="pf-phase-pill">PHASE 58</span>Phase 58 — Final V2 Results Summary</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V2 FINAL CLOSURE</span><span class="pf-meta-chip">Cells 147–148</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 147–148 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">332dd9c8</code> → <code class="pf-inline-code">9f864b91</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/results_rebuild.py">renderer</a> — <code class="pf-inline-code">render_verified_phase_result(58, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/reporting/v2_final_analysis.py">v2_final_analysis.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/model_improvement_v2/final_reporting_analysis/phase58_final_summary.json">phase58_final_summary.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-green">V2 FINAL</code>.</li>
</ul>
<p><strong>Current V2 presentation:</strong> Final V2 summary từ final lock, Step 17, MAPE và Phase 48–51; V1 attention được tách riêng.</p>
<p><strong>Historical V1 final-tables workflow context (provenance only):</strong></p>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/analysis/final_tables/{builders, writers, orchestrator, ...}.py</code></p>
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
<td>All analysis phases 48-57</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/final_tables/"><code class="pf-inline-code">artifacts/final_tables/</code></a><code class="pf-inline-code">{final_tables_manifest.json, final_tables_summary.json, final_tables_report.md, final_table_inventory.json, final_table_checksums.json, final_table_render_config.json, final_figure_inventory.json}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/final_tables/phase_58_signoff.json"><code class="pf-inline-code">phase_58_signoff.json</code></a></td>
</tr>
<tr>
<td><strong>Output (tables)</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/final_tables/tables/"><code class="pf-inline-code">artifacts/final_tables/tables/</code></a><code class="pf-inline-code">{latex,markdown,metadata}/</code></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>FT01-FT10: Final numerical tables</li>
<li>FA01-FA12: Final appendix tables</li>
<li>Render in LaTeX + Markdown formats</li>
<li>Metadata + checksums</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-phase-card pf-lineage-v2closure" data-phase="59"><h2 class="pf-heading pf-h2" id="phase-59-final-v2-conclusions"><span aria-hidden="true" class="pf-phase-pill">PHASE 59</span>Phase 59 — Final V2 Conclusions</h2><div class="pf-phase-meta"><span class="pf-meta-chip pf-meta-lineage">V2 FINAL CLOSURE</span><span class="pf-meta-chip">Cells 149–150</span></div>
<p><strong>Current notebook/reporting:</strong></p>
<ul>
<li>Notebook: Cells 149–150 · <a class="pf-link pf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Cell range/IDs identify the target; this link does not guarantee a jump to a specific notebook cell.">Open Notebook ↗</a> — IDs <code class="pf-inline-code">251da856</code> → <code class="pf-inline-code">76de7c69</code>.</li>
<li>Renderer: <a class="pf-link" href="../../src/course_work/reporting/results_rebuild.py">renderer</a> — <code class="pf-inline-code">render_verified_phase_result(59, PROJECT_ROOT)</code>.</li>
<li>Implementation/reporting source: <a class="pf-link" href="../../src/course_work/reporting/v2_final_analysis.py">v2_final_analysis.py</a>.</li>
<li>Verified artifact: <a class="pf-link" href="../../artifacts/model_improvement_v2/final_reporting_analysis/phase59_final_conclusions.json">phase59_final_conclusions.json</a>.</li>
<li>Lineage: <code class="pf-inline-code pf-code-green">V2 FINAL</code>.</li>
</ul>
<p><strong>Current V2 presentation:</strong> Kết luận final cho <code class="pf-inline-code">V2 Equal-weight Ensemble = FINAL_LOCKED_POLICY</code>; <code class="pf-inline-code">attention_conclusion_for_v2=NOT_CLAIMED</code>.</p>
<p><strong>Historical V1 final-conclusions workflow context (provenance only):</strong></p>
<p><strong>Module:</strong> <code class="pf-inline-code">src/course_work/analysis/final_conclusions/{builders, findings, writers, ...}.py</code></p>
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
<td>Phase 58 + all upstream</td>
</tr>
<tr>
<td><strong>Output</strong></td>
<td><a class="pf-link pf-artifact-link" href="../../artifacts/final_conclusions/"><code class="pf-inline-code">artifacts/final_conclusions/</code></a><code class="pf-inline-code">{final_conclusions_manifest.json, FINAL_PROJECT_SUMMARY.md}, final_submission_conclusion_package/{final_*.md}</code>, <a class="pf-link pf-artifact-link" href="../../artifacts/final_conclusions/phase_59_signoff.json"><code class="pf-inline-code">phase_59_signoff.json</code></a></td>
</tr>
</tbody>
</table></div>
<p><strong>Nhiệm vụ:</strong></p>
<ul>
<li>Final abstract results summary</li>
<li>Final conclusion section</li>
<li>Final research question answers</li>
<li>Final key takeaways</li>
<li>Final limitations</li>
<li>Final future work</li>
<li>Final viva defense notes</li>
<li>Final short conclusion</li>
</ul>
<hr class="pf-rule"/>
</section><section class="pf-section pf-summary-card"><h2 class="pf-heading pf-h2" id="tổng-kết-phases-34-59">Tổng Kết Phases 34-59</h2>
<div class="pf-table-wrap"><table class="pf-table">
<thead>
<tr>
<th>Nhóm</th>
<th>Số phases</th>
<th>Đặc điểm</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Sweeps 2</strong> (34-41)</td>
<td>8</td>
<td>S12-S19 hyperparameter sweeps</td>
</tr>
<tr>
<td><strong>V1 Candidate &amp; Lock</strong> (42–46)</td>
<td>5</td>
<td>Historical candidate synthesis → V1 model lock/runs</td>
</tr>
<tr>
<td><strong>Final presentation</strong> (47–59)</td>
<td>13</td>
<td>V2 47–51, V1 historical attention 52–57, V2 58–59</td>
</tr>
</tbody>
</table></div>
<hr class="pf-rule"/>
</section><section class="pf-section pf-structure-card"><h2 class="pf-heading pf-h2" id="cấu-trúc-phân-tầng-sau-refactor">Cấu Trúc Phân Tầng Sau Refactor</h2>
<p>Các module dưới đây là historical V1 analysis implementation. Current V2 Phase 48–51/58–59 reporting nằm tại <code class="pf-inline-code">course_work.reporting.v2_final_analysis</code> và không overwrite các artifact này:</p>
<pre class="pf-code"><code>analysis/
├── prediction_analysis/         # 48
├── residual_analysis/           # 49
├── error_regime_analysis/       # 50
├── worst_error_analysis/        # 51
├── attention_extraction/        # 52
├── attention_heatmaps/          # 53
├── last_query_attention/        # 54
├── head_comparison/             # 55
├── error_conditioned_attention/ # 56
├── seed_stability_attention/    # 57
├── final_tables/                # 58
└── final_conclusions/           # 59
</code></pre>
<p>Final execution phases:</p>
<ul>
<li><code class="pf-inline-code">course_work.final_model_lock</code> (45)</li>
<li><code class="pf-inline-code">course_work.final_test_evaluation</code> (47)</li>
</ul>
<p>Các phase runner scripts trong <code class="pf-inline-code">course_work/scripts/</code>:</p>
<pre class="pf-code"><code>src/course_work/scripts/
├── p39_*.py    (4 scripts)
├── p40_*.py    (3 scripts)
├── p42_p43_p44_p45_p46_p47_*.py
└── p47_*.py    (2 scripts)
</code></pre>
<hr class="pf-rule"/>
<p><strong>Phiên bản:</strong> 16/09/2026 — baseline chi tiết được giữ và đồng bộ với current verified flow.</p>
</section></div>
<div class="pf-handoff">
<div class="pf-handoff-grid">
<div aria-hidden="true" class="pf-handoff-icon">✓</div>
<div><h3>Flow closure</h3><p>Phase 59 closes the V2 reporting path while keeping V1 historical attention evidence explicitly separated from final V2 claims.</p></div>
<div aria-hidden="true" class="pf-handoff-arrow">→</div>
</div>
</div>
<div class="pf-footer">
<h3>End of Current Flow · Phases 34–59</h3>
<p>Recovered evidence → historical provenance → final V2 reporting → explicit closure.</p>
<div aria-hidden="true" class="pf-floaters">
<span class="pf-floater"><svg aria-hidden="true" viewbox="0 0 64 64"><path d="M31 5c-13 0-12 6-12 6v7h13v2H14S5 19 5 32s8 12 8 12h6v-8s0-8 8-8h13s7 0 7-7V12s1-7-16-7Zm-7 7a3 3 0 1 1 0 6 3 3 0 0 1 0-6Z" fill="#3776AB"></path><path d="M33 59c13 0 12-6 12-6v-7H32v-2h18s9 1 9-12-8-12-8-12h-6v8s0 8-8 8H24s-7 0-7 7v9s-1 7 16 7Zm7-7a3 3 0 1 1 0-6 3 3 0 0 1 0 6Z" fill="#FFD343"></path></svg></span>
<span class="pf-floater">🔥</span>
<span class="pf-floater"><svg aria-hidden="true" viewbox="0 0 64 64"><circle cx="14" cy="16" fill="#0b63ce" r="5"></circle><circle cx="32" cy="10" fill="#f5c84b" r="5"></circle><circle cx="50" cy="16" fill="#0b63ce" r="5"></circle><circle cx="20" cy="34" fill="#4cc9f0" r="5"></circle><circle cx="44" cy="34" fill="#4cc9f0" r="5"></circle><circle cx="32" cy="52" fill="#f5c84b" r="5"></circle><path d="M18 18l10-6m8 0 10 6M17 20l2 9m28-9-2 9M24 35h16m-17 4 6 9m12-9-6 9" fill="none" stroke="#334155" stroke-linecap="round" stroke-width="3"></path></svg></span>
<span class="pf-floater">📈</span>
<span class="pf-floater">✓</span>
</div>
<svg aria-hidden="true" class="pf-wave" preserveaspectratio="none" viewbox="0 0 1440 120">
<path d="M0,64 C240,118 410,8 720,58 C1000,105 1190,16 1440,72 L1440,120 L0,120 Z" fill="#8fd3ff" opacity=".50"></path>
<path d="M0,86 C220,34 470,120 760,76 C1030,34 1200,112 1440,58 L1440,120 L0,120 Z" fill="#f5c84b" opacity=".48"></path>
</svg>
</div>
</div>
