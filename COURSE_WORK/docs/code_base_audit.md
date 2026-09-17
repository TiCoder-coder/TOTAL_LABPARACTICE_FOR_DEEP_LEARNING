<!-- Visual refactor: CSS/SVG only; no JavaScript. Original audit content and source links preserved. -->
<style>
:root{--cba-ink:#0f172a;--cba-muted:#475467;--cba-blue:#0b63ce;--cba-blue2:#2f80ed;--cba-cyan:#38bdf8;--cba-yellow:#f7c948;--cba-yellow-soft:#fff7cc;--cba-paper:#fff;--cba-soft:#f6faff;--cba-line:#d8e7f5;--cba-green:#0f9f6e;--cba-green-soft:#e9fbf4;--cba-red:#c2410c;--cba-purple:#7c3aed}
*{box-sizing:border-box}html{scroll-behavior:smooth}.cba-shell{max-width:1280px;margin:0 auto;padding:24px 20px 0;background:linear-gradient(180deg,#fff 0%,#f7fbff 44%,#fffdf3 100%);color:var(--cba-ink);font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;line-height:1.72}
.cba-hero{position:relative;isolation:isolate;overflow:hidden;border:1px solid #cfe2f7;border-radius:30px;padding:42px 40px 34px;background:radial-gradient(circle at 87% 13%,rgba(247,201,72,.35),transparent 24%),radial-gradient(circle at 9% 9%,rgba(56,189,248,.24),transparent 29%),linear-gradient(135deg,#fff 0%,#f2f8ff 58%,#fff9da 100%);box-shadow:0 22px 65px rgba(15,81,145,.12)}.cba-hero:before,.cba-hero:after{content:"";position:absolute;border-radius:50%;z-index:-1}.cba-hero:before{width:260px;height:260px;right:-90px;bottom:-120px;border:1px solid rgba(11,99,206,.2);animation:cbaPulse 5s ease-in-out infinite}.cba-hero:after{width:130px;height:130px;right:-20px;bottom:-42px;border:1px solid rgba(247,201,72,.62);animation:cbaPulse 4s ease-in-out infinite reverse}.cba-kicker{font-size:12px;font-weight:900;letter-spacing:.18em;text-transform:uppercase;color:#0758b3}.cba-title{margin:7px 0 10px!important;color:#0b1220!important;font-size:clamp(34px,5vw,57px)!important;line-height:1.04!important;border:0!important}.cba-subtitle{max-width:940px;margin:0;color:#344054;font-size:16px}.cba-subtitle strong{color:#0b63ce}
.cba-badges{display:flex;flex-wrap:wrap;gap:10px;margin-top:22px}.cba-badge{display:inline-flex;align-items:center;gap:8px;padding:8px 12px;border:1px solid #d8e7f6;border-radius:999px;background:rgba(255,255,255,.92);font-size:13px;font-weight:760;box-shadow:0 6px 18px rgba(15,81,145,.08);transition:transform .22s ease,box-shadow .22s ease}.cba-badge:hover{transform:translateY(-3px);box-shadow:0 10px 24px rgba(15,81,145,.14)}.cba-badge svg{width:19px;height:19px;display:block}
.cba-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:22px}.cba-stat{padding:14px 15px;border-radius:16px;background:rgba(255,255,255,.86);border:1px solid #dce9f6;backdrop-filter:blur(4px)}.cba-stat-value{font-size:22px;font-weight:900;line-height:1.05;color:#0b63ce}.cba-stat:nth-child(3) .cba-stat-value{color:#7c3aed}.cba-stat:nth-child(4) .cba-stat-value{color:#087a57}.cba-stat-label{margin-top:6px;font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#667085}
.cba-hero-flow{display:grid;grid-template-columns:1fr auto 1fr auto 1fr auto 1fr auto 1fr;gap:8px;align-items:center;margin-top:22px}.cba-flow-node{padding:12px 10px;border-radius:14px;border:1px solid #d8e6f4;background:#fff;font-weight:800;text-align:center;box-shadow:0 5px 16px rgba(15,81,145,.06)}.cba-flow-node small{display:block;margin-top:2px;color:#667085;font-weight:650}.cba-flow-arrow{font-size:20px;color:#0b63ce;animation:cbaArrow 1.8s ease-in-out infinite}
.cba-document{padding:10px 0 40px}.cba-source-title{display:none}.cba-section{position:relative;margin:20px 0;padding:25px 27px;background:rgba(255,255,255,.97);border:1px solid var(--cba-line);border-radius:21px;box-shadow:0 8px 28px rgba(15,81,145,.065);transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease}.cba-section:hover{transform:translateY(-2px);box-shadow:0 15px 36px rgba(15,81,145,.105);border-color:#bdd9f3}.cba-heading{scroll-margin-top:20px;color:#101828}.cba-h1{display:none}.cba-h2{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:0 0 15px!important;padding:0 0 12px;border-bottom:3px solid transparent;border-image:linear-gradient(90deg,var(--cba-blue),var(--cba-cyan),var(--cba-yellow),transparent) 1;font-size:27px!important;line-height:1.25!important}.cba-h3{margin:25px 0 10px!important;color:#0b5eb8!important;font-size:20px!important}.cba-h3:before{content:"◆";margin-right:8px;color:var(--cba-yellow);font-size:.72em}.cba-section-pill{display:inline-flex;align-items:center;padding:5px 8px;border-radius:8px;background:#eaf4ff;color:#0758b3;font-size:10px;font-weight:900;letter-spacing:.1em;white-space:nowrap}.cba-anomaly-card .cba-section-pill{background:#fff2cc;color:#7d5700}.cba-repro-card .cba-section-pill,.cba-contract-card .cba-section-pill{background:#eafbf4;color:#087a57}.cba-phase-card .cba-section-pill{background:#f2eafe;color:#6d28d9}
.cba-toc-card{background:linear-gradient(135deg,#fbfdff,#f3f9ff 68%,#fffaf0)}.cba-toc-card:after{content:"Engineering Map";position:absolute;top:18px;right:22px;padding:5px 9px;border-radius:999px;background:#fff3b8;color:#7c5700;font-size:10px;font-weight:900;letter-spacing:.08em;text-transform:uppercase}.cba-toc-list{columns:2;column-gap:34px}.cba-toc-list>li{break-inside:avoid;margin:8px 0}.cba-project-card{background:radial-gradient(circle at 96% 8%,rgba(56,189,248,.16),transparent 28%),#fff}.cba-stack-card,.cba-structure-card{background:linear-gradient(135deg,#fff,#f7fbff)}.cba-architecture-card{background:linear-gradient(135deg,#fff,#f5faff 70%,#fffaf0)}.cba-contract-card{background:radial-gradient(circle at 96% 7%,rgba(15,159,110,.12),transparent 26%),#fff}.cba-artifact-card{background:radial-gradient(circle at 96% 5%,rgba(247,201,72,.17),transparent 28%),#fff}.cba-flow-card{background:linear-gradient(135deg,#fff,#f7fbff 68%,#fffaf1)}.cba-design-card{background:radial-gradient(circle at 96% 6%,rgba(124,58,237,.09),transparent 24%),#fff}.cba-anomaly-card{background:linear-gradient(135deg,#fffdf4,#fff 52%,#f8fbff)}.cba-appendix-card{background:linear-gradient(135deg,#fff,#f8fbff 60%,#fff9de)}
.cba-lineage-strip{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:18px 0}.cba-lineage-strip>div{position:relative;padding:14px 15px 14px 42px;border:1px solid #d9e7f4;border-radius:15px;background:#fff;box-shadow:0 6px 18px rgba(15,81,145,.05)}.cba-lineage-dot{position:absolute;left:16px;top:18px;width:12px;height:12px;border-radius:50%;box-shadow:0 0 0 5px rgba(11,99,206,.08)}.cba-lineage-dot.current{background:#10b981}.cba-lineage-dot.historical{background:#f7c948}.cba-lineage-dot.locked{background:#0b63ce}.cba-lineage-strip strong{display:block;font-size:13px}.cba-lineage-strip small{display:block;color:#667085;margin-top:2px}
.cba-nav-notice{display:grid;grid-template-columns:auto 1fr;gap:14px;align-items:start;margin:18px 0;padding:17px 18px;border-radius:16px;border:1px solid #cfe2f7;background:linear-gradient(90deg,#edf6ff,#fffdf0);box-shadow:0 7px 20px rgba(15,81,145,.06)}.cba-nav-notice-icon{width:38px;height:38px;display:grid;place-items:center;border-radius:11px;background:#0b63ce;color:white;font-size:20px;font-weight:900}.cba-nav-notice strong{color:#0758b3}.cba-nav-notice p{margin:3px 0 0;color:#475467}.cba-notebook-link{display:inline-flex!important;align-items:center;padding:2px 8px;border-radius:999px;background:#eaf4ff;border:1px solid #cfe2f7;text-decoration:none!important;font-size:.92em}.cba-notebook-link:hover{background:#ddecff;text-decoration:none!important}
.cba-note{margin:18px 0!important;padding:16px 18px!important;border-left:4px solid var(--cba-blue)!important;background:linear-gradient(90deg,#edf6ff,#fffdf2)!important;border-radius:0 13px 13px 0;color:#26364a}.cba-note:nth-of-type(2){border-left-color:#0f9f6e!important;background:linear-gradient(90deg,#eafbf4,#fffdf2)!important}.cba-rule{border:0!important;height:1px!important;background:linear-gradient(90deg,transparent,#93c5fd,#facc15,transparent)!important;margin:28px 0!important}.cba-link{color:#075fbe!important;text-decoration:none!important;font-weight:650}.cba-link:hover{text-decoration:underline!important;text-decoration-thickness:1.5px!important}.cba-inline-code{padding:.14em .42em;border:1px solid #d9e7f4;border-radius:6px;background:#f3f8fc;color:#854d0e;font-size:.92em}.cba-code-blue{background:#eaf4ff!important;color:#0758b3!important;border-color:#c9def4!important}.cba-code-yellow{background:#fff6c9!important;color:#7b5600!important;border-color:#f0dc86!important}.cba-code-green{background:#eafbf4!important;color:#087a57!important;border-color:#bdebd9!important}.cba-code{overflow:auto;padding:17px 18px;border:1px solid #cfe0ef;border-radius:14px;background:#f7fbff;box-shadow:inset 4px 0 0 #4cc9f0;white-space:pre}.cba-code code{background:transparent!important;color:#111827!important}.cba-flow-card .cba-code,.cba-architecture-card .cba-code{background:linear-gradient(135deg,#f7fbff,#fffdf5);box-shadow:inset 4px 0 0 #0b63ce}
.cba-table-wrap{overflow-x:auto;margin:18px 0;border:1px solid #d7e6f4;border-radius:14px;box-shadow:0 5px 18px rgba(15,81,145,.06)}.cba-table{width:100%;border-collapse:collapse;background:#fff;font-size:13.5px}.cba-table th{padding:11px 12px;background:linear-gradient(90deg,#eaf5ff,#fff6c9);color:#172033;text-align:left;border-bottom:1px solid #cbddeb}.cba-table td{padding:10px 12px;border-bottom:1px solid #edf2f7;vertical-align:top}.cba-table tr:last-child td{border-bottom:0}.cba-table tbody tr:hover{background:#f8fcff}.cba-anomaly-row td:first-child{font-weight:900;color:#9a6700}.cba-anomaly-row{background:linear-gradient(90deg,#fffdf4,#fff)}.cba-v2-row{background:linear-gradient(90deg,#f1fff9,#fff)!important}ul,ol{padding-left:24px}li::marker{color:#0b63ce}strong{color:#111827}
.cba-footer{position:relative;overflow:hidden;margin:24px -20px 0;padding:66px 24px 98px;text-align:center;background:linear-gradient(180deg,#f8fcff,#e9f6ff 54%,#fff4bf);border-top:1px solid #cfe4f7}.cba-footer h3{margin:0;color:#101828;font-size:25px}.cba-footer p{margin:7px auto 0;max-width:780px;color:#475467}.cba-floaters{display:flex;justify-content:center;gap:18px;margin:23px 0 4px}.cba-floater{font-size:29px;display:inline-block;animation:cbaFloat 3s ease-in-out infinite}.cba-floater:nth-child(2){animation-delay:.3s}.cba-floater:nth-child(3){animation-delay:.6s}.cba-floater:nth-child(4){animation-delay:.9s}.cba-floater:nth-child(5){animation-delay:1.2s}.cba-wave{position:absolute;left:-1%;right:-1%;bottom:-3px;width:102%;height:82px;opacity:.6}.cba-wave path:first-child{animation:cbaWave 6s ease-in-out infinite alternate}.cba-wave path:last-child{animation:cbaWave 7s ease-in-out infinite alternate-reverse}
@keyframes cbaFloat{0%,100%{transform:translateY(0) rotate(0)}50%{transform:translateY(-10px) rotate(2deg)}}@keyframes cbaPulse{0%,100%{transform:scale(1);opacity:.75}50%{transform:scale(1.08);opacity:1}}@keyframes cbaArrow{0%,100%{transform:translateX(0)}50%{transform:translateX(4px)}}@keyframes cbaWave{from{transform:translateX(-8px)}to{transform:translateX(8px)}}
@media (max-width:900px){.cba-stats{grid-template-columns:repeat(2,1fr)}.cba-hero-flow{grid-template-columns:1fr}.cba-flow-arrow{transform:rotate(90deg)}.cba-lineage-strip{grid-template-columns:1fr}.cba-toc-list{columns:1}.cba-hero{padding:30px 24px}.cba-section{padding:21px 20px}}@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}.cba-hero:before,.cba-hero:after,.cba-flow-arrow,.cba-floater,.cba-wave path{animation:none!important}}
</style>
<div class="cba-shell">
<div class="cba-hero">
<div class="cba-kicker">COURSE_WORK · ENGINEERING AUDIT · CURRENT VERIFIED UPDATE</div>
<h1 class="cba-title">Code Base Audit</h1>
<p class="cba-subtitle">Deep Learning Coursework engineering handbook: <strong>architecture, contracts, artifacts, phases, reproducibility, anomalies and V1/V2 lineage</strong> — with the original 06/09/2026 audit preserved and the 17/09/2026 repository mapping surfaced clearly.</p>
<div class="cba-badges"><span class="cba-badge"><svg viewBox="0 0 48 48" aria-hidden="true"><path fill="#3776AB" d="M23.7 4c-9 0-8.4 3.9-8.4 3.9v4h8.6v1.2H12S5.8 12.4 5.8 22.2s5.4 9.5 5.4 9.5h3.2v-4.5s-.2-5.4 5.3-5.4h8.8s4.9.1 4.9-4.8V9.2S34.1 4 23.7 4zm-4.8 3.1a1.6 1.6 0 1 1 0 3.2 1.6 1.6 0 0 1 0-3.2z"/><path fill="#FFD343" d="M24.3 44c9 0 8.4-3.9 8.4-3.9v-4h-8.6v-1.2H36s6.2.7 6.2-9.1-5.4-9.5-5.4-9.5h-3.2v4.5s.2 5.4-5.3 5.4h-8.8s-4.9-.1-4.9 4.8v7.8S13.9 44 24.3 44zm4.8-3.1a1.6 1.6 0 1 1 0-3.2 1.6 1.6 0 0 1 0 3.2z"/></svg> Python 3.10</span><span class="cba-badge">🔥 PyTorch</span><span class="cba-badge">◫ Jupyter</span><span class="cba-badge">⚡ Transformer</span><span class="cba-badge">⌁ Time Series</span><span class="cba-badge">◈ Clean Architecture</span><span class="cba-badge">✓ SHA-256 Governance</span></div>
<div class="cba-stats"><div class="cba-stat"><div class="cba-stat-value">60</div><div class="cba-stat-label">Phases · 0–59</div></div><div class="cba-stat"><div class="cba-stat-value">153</div><div class="cba-stat-label">Notebook cells</div></div><div class="cba-stat"><div class="cba-stat-value">430</div><div class="cba-stat-label">Python files</div></div><div class="cba-stat"><div class="cba-stat-value">V2 FINAL</div><div class="cba-stat-label">Current presentation</div></div></div>
<div class="cba-hero-flow"><div class="cba-flow-node">Contract<small>Single source of truth</small></div><div class="cba-flow-arrow">→</div><div class="cba-flow-node">Data & Models<small>Phases 1–22</small></div><div class="cba-flow-arrow">→</div><div class="cba-flow-node">Sweeps<small>S1–S19</small></div><div class="cba-flow-arrow">→</div><div class="cba-flow-node">Final Governance<small>Lock · Refit · Benchmark</small></div><div class="cba-flow-arrow">→</div><div class="cba-flow-node">Reporting<small>V2 + historical provenance</small></div></div>
</div>
<div class="cba-document"><h1 class="cba-heading cba-h1 cba-source-title" id="code-base-audit--deep-learning-coursework">CODE BASE AUDIT — Deep Learning Coursework</h1>
<blockquote class="cba-note">
<p><strong>Audit Timestamp:</strong> 06/09/2026 (UTC+7)
<strong>Repository:</strong> <a class="cba-link" href="../../COURSE_WORK/"><code class="cba-inline-code">COURSE_WORK</code></a>
<strong>Audit Scope:</strong> Toàn bộ luồng, code, follow, structure, conventions, design patterns, anomalies của project COURSE_WORK
<strong>Mục đích:</strong> Tài liệu duy nhất để hiểu project, onboarding developer mới, đánh giá độ chuẩn chỉnh của codebase</p>
</blockquote>
<hr class="cba-rule"/>
<blockquote class="cba-note">
<p><strong>CURRENT VERIFIED UPDATE — 17/09/2026</strong></p>
<p>Phần kiến trúc chi tiết của audit gốc được bảo toàn. Các dữ kiện thay đổi theo repo hiện tại đã được cập nhật: notebook có <code class="cba-inline-code">153</code> cells; Phase 34–41 là <code class="cba-inline-code cba-code-blue">V1 RECOVERED / VALID_REUSABLE</code>; Phase 47–51 và 58–59 là <code class="cba-inline-code cba-code-green">V2 FINAL</code>; Phase 52–57 là <code class="cba-inline-code cba-code-blue">V1 HISTORICAL ATTENTION</code>. Final policy là <code class="cba-inline-code cba-code-green">MEAN_E14_M1_SEEDS_42_123_2026</code> với equal weights <code class="cba-inline-code">[1/3,1/3,1/3]</code>. Step 17 mang nhãn <code class="cba-inline-code cba-code-green">POST_HOC_V2_BENCHMARK</code>, không phải unseen unbiased Test.</p>
<p><a class="cba-link cba-notebook-link" href="../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Markdown links do not reliably deep-link to an exact notebook cell.">Open Notebook ↗</a> · <a class="cba-link" href="link&amp;discussion_to_result/NOTEBOOK_CELLS_WALKTHROUGH.md">Cell walkthrough</a> · <a class="cba-link" href="current_flow/CURRENT_FLOW_SUMMARY.md">Current flow</a> · <a class="cba-link" href="../artifacts/model_improvement_v2/final_model_lock/v2_final_model_lock.json">V2 final lock</a> · <a class="cba-link" href="../artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics_with_mape.json">Step 17 + MAPE</a></p>
</blockquote>
<blockquote class="cba-note"><p><strong>Clickable path policy:</strong> file/directory chips có link chỉ được tạo khi target tồn tại trong current repository. Glob, placeholder, schema example và historical missing path vẫn giữ dạng code thuần để không tạo liên kết giả.</p></blockquote>
<section class="cba-section"><h2 class="cba-heading cba-h2" id="-mục-lục"><span class="cba-section-pill">SECTION 1</span>📑 Mục Lục</h2>
<ol>
<li><a class="cba-link" href="#1-tổng-quan-project">Tổng Quan Project</a></li>
<li><a class="cba-link" href="#2-tech-stack--môi-trường">Tech Stack &amp; Môi Trường</a></li>
<li><a class="cba-link" href="#3-cấu-trúc-thư-mục">Cấu Trúc Thư Mục</a></li>
<li><a class="cba-link" href="#4-source-code-organization">Source Code Organization</a></li>
<li><a class="cba-link" href="#5-configuration--contracts">Configuration &amp; Contracts</a></li>
<li><a class="cba-link" href="#6-tests-layer">Tests Layer</a></li>
<li><a class="cba-link" href="#7-artifacts-layer">Artifacts Layer</a></li>
<li><a class="cba-link" href="#8-data-layer">Data Layer</a></li>
<li><a class="cba-link" href="#9-notebook-layer">Notebook Layer</a></li>
<li><a class="cba-link" href="#10-documentation-layer">Documentation Layer</a></li>
<li><a class="cba-link" href="#11-phase-mapping-chi-tiết-0-59">Phase Mapping Chi Tiết (0-59)</a></li>
<li><a class="cba-link" href="#12-data-flow-tổng-quan">Data Flow Tổng Quan</a></li>
<li><a class="cba-link" href="#13-design-patterns--solid">Design Patterns &amp; SOLID</a></li>
<li><a class="cba-link" href="#14-reproducibility-story">Reproducibility Story</a></li>
<li><a class="cba-link" href="#15-key-python-modules---detailed">Key Python Modules - Detailed</a></li>
<li><a class="cba-link" href="#16-anomalies--observations">Anomalies &amp; Observations</a></li>
<li><a class="cba-link" href="#17-quy-ước--rule-base">Quy Ước &amp; Rule Base</a></li>
<li><a class="cba-link" href="#18-phụ-lục">Phụ Lục</a></li>
</ol>
<hr class="cba-rule"/>
</section><section class="cba-section cba-project-card"><h2 class="cba-heading cba-h2" id="1-tổng-quan-project"><span class="cba-section-pill">PROJECT</span>1. Tổng Quan Project</h2>
<h3 class="cba-heading cba-h3" id="11-mục-tiêu">1.1. Mục tiêu</h3>
<p>Project Deep Learning cho <strong>Multivariate Time-Series Regression</strong> trên dataset <strong>UCI Appliances Energy Prediction</strong>:</p>
<ul>
<li><strong>Input:</strong> Cửa sổ lookback (36/72/144 step = 6h/12h/24h) chứa 29 biến cảm biến (nhiệt độ, độ ẩm, ánh sáng, ...).</li>
<li><strong>Target:</strong> <code class="cba-inline-code">Appliances</code> (Wh) tại step <code class="cba-inline-code">t+1</code> (10 phút sau).</li>
<li><strong>Sample definition:</strong> $X[t-L+1:t] \rightarrow Appliances[t+1]$.</li>
<li><strong>Models:</strong> Persistence, LSTM, Transformer Encoder.</li>
<li><strong>Evaluation:</strong> MAE, RMSE, R² ở đơn vị Wh gốc (không scaled space).</li>
<li><strong>Split:</strong> Chronological 70/15/15 (TRAIN / VALIDATION / TEST).</li>
<li><strong>Final seeds:</strong> 42, 123, 2026.</li>
</ul>
<h3 class="cba-heading cba-h3" id="12-quy-mô-project">1.2. Quy mô project</h3>
<div class="cba-table-wrap"><table class="cba-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Quy mô</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Phases</strong></td>
<td>60 (0-59, đánh số 0-based cho environment, 1-based cho data)</td>
</tr>
<tr>
<td><strong>Source code</strong></td>
<td>430 file Python, 155,204 LOC</td>
</tr>
<tr>
<td><strong>Tests</strong></td>
<td>149 file Python, 41,622 LOC</td>
</tr>
<tr>
<td><strong>Notebook</strong></td>
<td>153 cells</td>
</tr>
<tr>
<td><strong>Artifacts</strong></td>
<td>49 top-level directories</td>
</tr>
<tr>
<td><strong>Docs</strong></td>
<td>3 current_flow + 1 notebook walkthrough + 1 audit (file này) + 79 analysis_error + 112 save logs</td>
</tr>
<tr>
<td><strong>Scripts</strong></td>
<td>76 files ở root + 26 in-package</td>
</tr>
</tbody>
</table></div>
<h3 class="cba-heading cba-h3" id="13-đặc-điểm-nổi-bật">1.3. Đặc điểm nổi bật</h3>
<ul>
<li><strong>Single Source of Truth:</strong> Một file <a class="cba-link cba-path-link" href="../configs/base/coursework_contract.json"><code class="cba-inline-code">configs/base/coursework_contract.json</code></a> quy định toàn bộ.</li>
<li><strong>Test Firewall:</strong> Trong V1 historical contract, Test bị khóa đến Phase 47. Trong V2, Test chỉ được mở sau Step 16 final lock tại Step 17 <code class="cba-inline-code cba-code-green">POST_HOC_V2_BENCHMARK</code>; không dùng để retune.</li>
<li><strong>Phase Gates:</strong> Mỗi phase tạo <code class="cba-inline-code">phase_N_signoff.json</code> với SHA-256, phase sau kiểm tra trước khi chạy.</li>
<li><strong>NO-TRAIN Phases:</strong> Phases 45, 47 chỉ lock/eval, không train.</li>
<li><strong>FINAL_REFIT mode:</strong> Phase 46 train over TRAIN+VAL (không hold out VAL).</li>
<li><strong>Fold-local scalers:</strong> Phase 44 fit scaler per-fold (không leak từ fold khác).</li>
<li><strong>Strict reproducibility:</strong> D0 mode, seed 42 mặc định, hash mọi artifact.</li>
</ul>
<hr class="cba-rule"/>
</section><section class="cba-section cba-stack-card"><h2 class="cba-heading cba-h2" id="2-tech-stack--môi-trường"><span class="cba-section-pill">STACK</span>2. Tech Stack &amp; Môi Trường</h2>
<h3 class="cba-heading cba-h3" id="21-python-target">2.1. Python Target</h3>
<ul>
<li><strong>Python:</strong> <code class="cba-inline-code">&gt;=3.10,&lt;3.11</code> (locked trong <a class="cba-link cba-path-link" href="../pyproject.toml"><code class="cba-inline-code">pyproject.toml</code></a>)</li>
<li><strong>Lý do:</strong> Tương thích tốt nhất với torch 2.13 và tất cả dependencies.</li>
</ul>
<h3 class="cba-heading cba-h3" id="22-pinned-dependencies--requirementstxt-">2.2. Pinned Dependencies (<a class="cba-link cba-path-link" href="../requirements.txt"><code class="cba-inline-code">requirements.txt</code></a>)</h3>
<div class="cba-table-wrap"><table class="cba-table">
<thead>
<tr>
<th>Package</th>
<th>Version</th>
<th>Vai trò</th>
</tr>
</thead>
<tbody>
<tr>
<td><code class="cba-inline-code">ipykernel</code></td>
<td>7.3.0</td>
<td>Jupyter kernel</td>
</tr>
<tr>
<td><code class="cba-inline-code">joblib</code></td>
<td>1.5.3</td>
<td>Scaler serialization</td>
</tr>
<tr>
<td><code class="cba-inline-code">jupyter</code></td>
<td>1.1.1</td>
<td>Notebook interface</td>
</tr>
<tr>
<td><code class="cba-inline-code">matplotlib</code></td>
<td>3.10.9</td>
<td>Visualization</td>
</tr>
<tr>
<td><code class="cba-inline-code">nbclient</code></td>
<td>0.11.0</td>
<td>Notebook execution</td>
</tr>
<tr>
<td><code class="cba-inline-code">nbformat</code></td>
<td>5.11.0</td>
<td>Notebook I/O</td>
</tr>
<tr>
<td><code class="cba-inline-code">numpy</code></td>
<td>2.2.6</td>
<td>Numerical ops</td>
</tr>
<tr>
<td><code class="cba-inline-code">pandas</code></td>
<td>2.3.3</td>
<td>DataFrames</td>
</tr>
<tr>
<td><code class="cba-inline-code">pytest</code></td>
<td>9.1.1</td>
<td>Testing framework</td>
</tr>
<tr>
<td><code class="cba-inline-code">scikit-learn</code></td>
<td>1.7.2</td>
<td>Scalers, metrics, ML utils</td>
</tr>
<tr>
<td><code class="cba-inline-code">torch</code></td>
<td>2.13.0</td>
<td>Deep learning core</td>
</tr>
<tr>
<td><code class="cba-inline-code">seaborn</code></td>
<td>0.13.2</td>
<td>Statistical viz</td>
</tr>
</tbody>
</table></div>
<h3 class="cba-heading cba-h3" id="23-device-selection">2.3. Device Selection</h3>
<p>Code dùng <code class="cba-inline-code">utils.environment.select_device()</code> với thứ tự ưu tiên:</p>
<ol>
<li><strong>CUDA</strong> (nếu có GPU NVIDIA)</li>
<li><strong>MPS</strong> (Apple Silicon)</li>
<li><strong>CPU</strong> (fallback)</li>
</ol>
<p>Current verified device: <strong>MPS</strong> trên Apple Silicon; code vẫn giữ thứ tự fallback CUDA → MPS → CPU.</p>
<h3 class="cba-heading cba-h3" id="24-determinism">2.4. Determinism</h3>
<ul>
<li><strong>Default mode:</strong> D0 (fully deterministic)</li>
<li><strong>DEVELOPMENT_SEED:</strong> 42</li>
<li><strong>FINAL_SEEDS:</strong> (42, 123, 2026)</li>
<li><strong>Configuration:</strong> <code class="cba-inline-code cba-code-blue">utils/reproducibility.configure_reproducibility(mode="D0"|"D1")</code></li>
</ul>
<hr class="cba-rule"/>
</section><section class="cba-section cba-structure-card"><h2 class="cba-heading cba-h2" id="3-cấu-trúc-thư-mục"><span class="cba-section-pill">STRUCTURE</span>3. Cấu Trúc Thư Mục</h2>
<h3 class="cba-heading cba-h3" id="31-top-level-layout">3.1. Top-Level Layout</h3>
<pre class="cba-code"><code>COURSE_WORK/
├── artifacts/            # 49 top-level directories - output của mỗi phase
├── configs/              # Single source of truth contract
│   └── base/
│       └── coursework_contract.json
├── docs/                 # Documentation layer
│   ├── analysis_error/   # 79 issue analyses
│   ├── code_base_audit.md   # File này
│   ├── current_flow/     # 3 files - current architecture
│   ├── link&amp;discussion_to_result/  # Notebook walkthrough
│   ├── plan/             # Historical &amp; current plans
│   ├── rule_base/        # Coding conventions
│   └── save_log_in_processing/  # 112 phase logs
├── link/                 # Symlink mirror của data/
│   ├── interim/
│   └── raw_data/
├── notebook_course_work/
│   └── CourseWork.ipynb  # Single notebook, 153 cells
├── scripts/              # 76 root-level scripts (~50 disposable probes)
├── src/                  # Python package
│   └── course_work/      # 22 subpackages
├── tests/                # 149 test files
├── pyproject.toml        # Package metadata
└── requirements.txt      # Pinned deps
</code></pre>
<h3 class="cba-heading cba-h3" id="32-srccoursework--source-package">3.2. <a class="cba-link cba-path-link" href="../src/course_work"><code class="cba-inline-code">src/course_work/</code></a> — Source Package</h3>
<p>22 subpackages, tổ chức theo <strong>Clean Architecture</strong>:</p>
<div class="cba-table-wrap"><table class="cba-table">
<thead>
<tr>
<th>Tầng</th>
<th>Subpackages</th>
<th>Trách nhiệm</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>CORE DOMAIN</strong></td>
<td><a class="cba-link cba-path-link" href="../src/course_work/models"><code class="cba-inline-code">models/</code></a>, <a class="cba-link cba-path-link" href="../src/course_work/training"><code class="cba-inline-code">training/</code></a>, <a class="cba-link cba-path-link" href="../src/course_work/attention"><code class="cba-inline-code">attention/</code></a>, <a class="cba-link cba-path-link" href="../src/course_work/evaluation"><code class="cba-inline-code">evaluation/</code></a></td>
<td>Neural architectures, training loop, metrics</td>
</tr>
<tr>
<td><strong>DATA PIPELINE</strong></td>
<td><a class="cba-link cba-path-link" href="../src/course_work/data"><code class="cba-inline-code">data/</code></a></td>
<td>ETL: acquisition → features → splits → scaling → windows → datasets</td>
</tr>
<tr>
<td><strong>CONFIG &amp; GOVERNANCE</strong></td>
<td><a class="cba-link cba-path-link" href="../src/course_work/contracts"><code class="cba-inline-code">contracts/</code></a>, <a class="cba-link cba-path-link" href="../src/course_work/sweeps"><code class="cba-inline-code">sweeps/</code></a>, <a class="cba-link cba-path-link" href="../src/course_work/sanity"><code class="cba-inline-code">sanity/</code></a>, <a class="cba-link cba-path-link" href="../src/course_work/diagnostics"><code class="cba-inline-code">diagnostics/</code></a>, <a class="cba-link cba-path-link" href="../src/course_work/lstm_tuning"><code class="cba-inline-code">lstm_tuning/</code></a>, <a class="cba-link cba-path-link" href="../src/course_work/rolling_origin"><code class="cba-inline-code">rolling_origin/</code></a>, <a class="cba-link cba-path-link" href="../src/course_work/metric_addendum"><code class="cba-inline-code">metric_addendum/</code></a>, <a class="cba-link cba-path-link" href="../src/course_work/experiments"><code class="cba-inline-code">experiments/</code></a>, <a class="cba-link cba-path-link" href="../src/course_work/verification"><code class="cba-inline-code">verification/</code></a></td>
<td>Phase orchestration &amp; validation</td>
</tr>
<tr>
<td><strong>INFRASTRUCTURE</strong></td>
<td><code class="cba-inline-code">utils/</code>, <code class="cba-inline-code">scaling/</code> (re-export)</td>
<td>I/O, environment, reproducibility</td>
</tr>
<tr>
<td><strong>EXECUTION (NO-TRAIN)</strong></td>
<td><code class="cba-inline-code">final_model_lock/</code>, <code class="cba-inline-code">final_test_evaluation/</code></td>
<td>Final governance</td>
</tr>
<tr>
<td><strong>ANALYSIS</strong></td>
<td><a class="cba-link cba-path-link" href="../src/course_work/analysis"><code class="cba-inline-code">analysis/</code></a></td>
<td>12 subpackages cho phases 48-59</td>
</tr>
<tr>
<td><strong>REPORTING</strong></td>
<td><a class="cba-link cba-path-link" href="../src/course_work/reporting"><code class="cba-inline-code">reporting/</code></a></td>
<td>HTML dashboards</td>
</tr>
<tr>
<td><strong>PIPELINE</strong></td>
<td><a class="cba-link cba-path-link" href="../scripts"><code class="cba-inline-code">scripts/</code></a></td>
<td>Phase orchestrators (in-package)</td>
</tr>
<tr>
<td><strong>BASELINES</strong></td>
<td><code class="cba-inline-code">baselines/</code></td>
<td>Persistence, LSTM_B0, Transformer_B0</td>
</tr>
</tbody>
</table></div>
<h3 class="cba-heading cba-h3" id="33-tests-structure">3.3. <a class="cba-link cba-path-link" href="../tests"><code class="cba-inline-code">tests/</code></a> Structure</h3>
<pre class="cba-code"><code>tests/
├── conftest.py           # Setup: adds src/ to sys.path, sets COURSE_WORK_ROOT
├── contracts/            # 2 files
│   ├── test_coursework_contract.py
│   └── test_selective_execution_policy.py
├── integration/          # 9 files - cross-phase end-to-end
│   ├── test_phase_0_to_5_chain.py
│   ├── test_phase_0_to_8_presentation.py
│   ├── test_notebook_boundary.py
│   ├── test_final_dev_dataset.py
│   ├── test_phase46_*.py  (3 files)
│   ├── test_phase47_infrastructure.py
│   ├── test_mape_addendum.py
│   └── dry_run_final_dev.py
└── unit/                 # ~97 files - per-phase/per-module
</code></pre>
<h3 class="cba-heading cba-h3" id="34-artifacts-structure-49-top-level-directories">3.4. <a class="cba-link cba-path-link" href="../artifacts"><code class="cba-inline-code">artifacts/</code></a> Structure (49 top-level directories)</h3>
<p>Mỗi phase có directory riêng với pattern:</p>
<ul>
<li><code class="cba-inline-code">phase_N_signoff.json</code> — phase gate (status: OK/FAIL)</li>
<li><code class="cba-inline-code">phase_N_manifest.json</code> — input/output checksums</li>
<li><code class="cba-inline-code">phase_N_contract.json</code> — config lock</li>
<li><code class="cba-inline-code">phase_N_report.md</code> — human-readable summary</li>
<li><code class="cba-inline-code">phase_N_discrepancies.json</code> — issues tracker</li>
<li><code class="cba-inline-code">figures/</code>, <code class="cba-inline-code">tables/</code> — visualization</li>
</ul>
<hr class="cba-rule"/>
</section><section class="cba-section cba-architecture-card"><h2 class="cba-heading cba-h2" id="4-source-code-organization"><span class="cba-section-pill">ARCHITECTURE</span>4. Source Code Organization</h2>
<h3 class="cba-heading cba-h3" id="41-module-dependency-graph">4.1. Module Dependency Graph</h3>
<pre class="cba-code"><code>                    ┌─────────────────┐
                    │   contracts     │  (coursework.py)
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   ┌─────────┐         ┌──────────┐         ┌──────────┐
   │  data/  │         │  utils/  │         │evaluation│
   └────┬────┘         └─────┬────┘         └────┬─────┘
        │                    │                    │
        └─────────┬──────────┴──────────┬─────────┘
                  │                     │
                  ▼                     ▼
            ┌──────────┐         ┌────────────┐
            │  models/ │         │experiments/│
            └─────┬────┘         └──────┬─────┘
                  │                     │
                  ▼                     │
            ┌──────────┐                │
            │ training/│                │
            └─────┬────┘                │
                  │                     │
                  ▼                     │
            ┌──────────┐                │
            │ baselines│────────────────┤
            └──────────┘                │
                                        │
        ┌───────────────────────────────┼─────────────────────┐
        │                               │                     │
        ▼                               ▼                     ▼
   ┌─────────┐                   ┌──────────┐          ┌───────────┐
   │ sweeps/ │                   │lstm_tuning│         │rolling_origin│
   └─────────┘                   └──────────┘          └───────┬───┘
                                                                │
        ┌───────────────────────────────────────────────────────┤
        │                                                       │
        ▼                                                       ▼
   ┌──────────────┐                                   ┌──────────────────┐
   │final_model_  │                                   │final_test_       │
   │lock/         │                                   │evaluation/       │
   └──────────────┘                                   └──────────────────┘
        │                                                       │
        └──────────────────┬────────────────────────────────────┘
                           ▼
                  ┌─────────────────┐
                  │    analysis/    │  (12 subpackages, 48-59)
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   reporting/    │  (HTML dashboards)
                  └─────────────────┘
</code></pre>
<h3 class="cba-heading cba-h3" id="42-quy-tắc-phân-tầng-strict">4.2. Quy Tắc Phân Tầng (Strict)</h3>
<ul>
<li><code class="cba-inline-code">models/</code> KHÔNG import <code class="cba-inline-code">data/</code>, <code class="cba-inline-code">training/</code>, <code class="cba-inline-code">experiments/</code>.</li>
<li><code class="cba-inline-code">data/</code> KHÔNG import <code class="cba-inline-code">models/</code>, <code class="cba-inline-code">training/</code>.</li>
<li><code class="cba-inline-code">evaluation/</code> chỉ phụ thuộc <code class="cba-inline-code">numpy</code>, <code class="cba-inline-code">pandas</code>, <code class="cba-inline-code">utils/</code> — pure functions.</li>
<li><code class="cba-inline-code">analysis/</code> chỉ đọc artifacts, KHÔNG mutate.</li>
<li><a class="cba-link cba-path-link" href="../scripts"><code class="cba-inline-code">scripts/</code></a> ở ngoài cùng, được phép import bất kỳ tầng nào.</li>
<li>Mọi I/O đi qua <code class="cba-inline-code">utils.artifacts</code> (single entry point).</li>
</ul>
<h3 class="cba-heading cba-h3" id="43-module-inventory">4.3. Module Inventory</h3>
<div class="cba-table-wrap"><table class="cba-table">
<thead>
<tr>
<th>Module</th>
<th>LOC</th>
<th>Files</th>
<th>Purpose</th>
</tr>
</thead>
<tbody>
<tr>
<td><a class="cba-link cba-path-link" href="../src/course_work/analysis"><code class="cba-inline-code">analysis/</code></a></td>
<td>n/a</td>
<td>12 packages + <code class="cba-inline-code">__init__.py</code></td>
<td>Read-only analysis phases 48-59</td>
</tr>
<tr>
<td><a class="cba-link cba-path-link" href="../src/course_work/attention"><code class="cba-inline-code">attention/</code></a></td>
<td>352</td>
<td>6</td>
<td>Attention contracts + Phase 17 verification</td>
</tr>
<tr>
<td><code class="cba-inline-code">baselines/</code></td>
<td>2,047</td>
<td>3</td>
<td>Persistence (P14), LSTM_B0 (P20), Transformer_B0 (P21)</td>
</tr>
<tr>
<td><a class="cba-link cba-path-link" href="../src/course_work/contracts"><code class="cba-inline-code">contracts/</code></a></td>
<td>183</td>
<td>1</td>
<td><a class="cba-link cba-path-link" href="../src/course_work/contracts/coursework.py"><code class="cba-inline-code">coursework.py</code></a> - Phase 0 contract loader</td>
</tr>
<tr>
<td><a class="cba-link cba-path-link" href="../src/course_work/data"><code class="cba-inline-code">data/</code></a></td>
<td>6,365</td>
<td>11</td>
<td>Phases 2-11 pipeline</td>
</tr>
<tr>
<td><code class="cba-inline-code">diagnostics/</code></td>
<td>1,030</td>
<td>2</td>
<td>Phase 22 learning diagnostics</td>
</tr>
<tr>
<td><a class="cba-link cba-path-link" href="../src/course_work/evaluation"><code class="cba-inline-code">evaluation/</code></a></td>
<td>1,182</td>
<td>1</td>
<td><a class="cba-link cba-path-link" href="../src/course_work/evaluation/metrics.py"><code class="cba-inline-code">metrics.py</code></a> - Phase 12 metrics</td>
</tr>
<tr>
<td><code class="cba-inline-code">experiments/</code></td>
<td>2,862</td>
<td>3</td>
<td><a class="cba-link cba-path-link" href="../src/course_work/experiments/registry.py"><code class="cba-inline-code">registry.py</code></a> - Phase 13 + sweep execution</td>
</tr>
<tr>
<td><code class="cba-inline-code">final_model_lock/</code></td>
<td>2,393</td>
<td>10</td>
<td>Phase 45 NO-TRAIN governance</td>
</tr>
<tr>
<td><code class="cba-inline-code">final_test_evaluation/</code></td>
<td>3,852</td>
<td>6</td>
<td>Phase 47 final test</td>
</tr>
<tr>
<td><code class="cba-inline-code">lstm_tuning/</code></td>
<td>1,785</td>
<td>9</td>
<td>Phase 43 multi-stage tuning</td>
</tr>
<tr>
<td><code class="cba-inline-code">metric_addendum/</code></td>
<td>460</td>
<td>5</td>
<td>MAPE addendum</td>
</tr>
<tr>
<td><a class="cba-link cba-path-link" href="../src/course_work/models"><code class="cba-inline-code">models/</code></a></td>
<td>1,440</td>
<td>6</td>
<td>LSTM, Transformer, RevIN, positional encoding</td>
</tr>
<tr>
<td><a class="cba-link cba-path-link" href="../src/course_work/reporting"><code class="cba-inline-code">reporting/</code></a></td>
<td>n/a</td>
<td>16</td>
<td>Per-phase HTML dashboards</td>
</tr>
<tr>
<td><code class="cba-inline-code">rolling_origin/</code></td>
<td>~9,700</td>
<td>22 (+tests)</td>
<td>Phase 44 rolling-origin robustness</td>
</tr>
<tr>
<td><code class="cba-inline-code">sanity/</code></td>
<td>287</td>
<td>1</td>
<td>Phase 18 forward sanity</td>
</tr>
<tr>
<td><code class="cba-inline-code">scaling/</code></td>
<td>907</td>
<td>1</td>
<td>Public re-export of <code class="cba-inline-code">data.scaling</code></td>
</tr>
<tr>
<td><a class="cba-link cba-path-link" href="../scripts"><code class="cba-inline-code">scripts/</code></a></td>
<td>~15,500</td>
<td>28</td>
<td>Phase 39-47 orchestrators</td>
</tr>
<tr>
<td><a class="cba-link cba-path-link" href="../src/course_work/sweeps"><code class="cba-inline-code">sweeps/</code></a></td>
<td>~11,000</td>
<td>12</td>
<td>Hyperparameter sweeps S1-S19</td>
</tr>
<tr>
<td><a class="cba-link cba-path-link" href="../src/course_work/training"><code class="cba-inline-code">training/</code></a></td>
<td>937</td>
<td>3</td>
<td><a class="cba-link cba-path-link" href="../src/course_work/training/engine.py"><code class="cba-inline-code">engine.py</code></a> - Phase 19</td>
</tr>
<tr>
<td><code class="cba-inline-code">utils/</code></td>
<td>714</td>
<td>3</td>
<td><a class="cba-link cba-path-link" href="../src/course_work/utils/artifacts.py"><code class="cba-inline-code">artifacts.py</code></a>, <a class="cba-link cba-path-link" href="../src/course_work/utils/environment.py"><code class="cba-inline-code">environment.py</code></a>, <a class="cba-link cba-path-link" href="../src/course_work/utils/reproducibility.py"><code class="cba-inline-code">reproducibility.py</code></a></td>
</tr>
<tr>
<td><code class="cba-inline-code">verification/</code></td>
<td>1,208</td>
<td>2</td>
<td>Phase 46 verification</td>
</tr>
</tbody>
</table></div>
<hr class="cba-rule"/>
</section><section class="cba-section cba-contract-card"><h2 class="cba-heading cba-h2" id="5-configuration--contracts"><span class="cba-section-pill">CONTRACTS</span>5. Configuration &amp; Contracts</h2>
<h3 class="cba-heading cba-h3" id="51-single-source-of-truth">5.1. Single Source of Truth</h3>
<p><strong>File:</strong> <a class="cba-link cba-path-link" href="../configs/base/coursework_contract.json"><code class="cba-inline-code">configs/base/coursework_contract.json</code></a>
<strong>Version:</strong> <code class="cba-inline-code">COURSEWORK-CONTRACT-v1</code>
<strong>Fingerprint:</strong> SHA-256 của canonical JSON</p>
<h3 class="cba-heading cba-h3" id="52-contract-schema">5.2. Contract Schema</h3>
<pre class="cba-code"><code class="language-json">{
  "contract_version": "COURSEWORK-CONTRACT-v1",
  "problem": {
    "task": "multivariate_time_series_regression",
    "dataset": "UCI Appliances Energy Prediction",
    "target": "Appliances",
    "target_unit": "Wh",
    "sampling_minutes": 10,
    "forecast_horizon_steps": 1,
    "sample_definition": "X[t-L+1:t] -&gt; Appliances[t+1]"
  },
  "lookbacks": {"options": [36, 72, 144], "primary": 144},
  "split": {"type": "chronological", "train": 0.7, "val": 0.15, "test": 0.15},
  "models": ["persistence", "lstm", "transformer_encoder"],
  "metrics": {"selection": "validation_rmse", "final": ["mae", "rmse", "r2"]},
  "final_seeds": [42, 123, 2026],
  "option_registry": { /* 18 groups, see below */ },
  "baseline_transformer": { /* reference config */ },
  "research_questions": [/* RQ1-RQ7 */],
  "protocol_violations": [/* 10 named violations */]
}
</code></pre>
<h3 class="cba-heading cba-h3" id="53-option-registry-18-groups">5.3. Option Registry (18 Groups)</h3>
<p>Mỗi group có explicit ID để sweep sử dụng:</p>
<div class="cba-table-wrap"><table class="cba-table">
<thead>
<tr>
<th>Group</th>
<th>IDs</th>
<th>Values</th>
</tr>
</thead>
<tbody>
<tr>
<td><code class="cba-inline-code">feature_sets</code></td>
<td>FS0, FS1, FS2</td>
<td>exogenous_only / +autoregressive / +random_controls</td>
</tr>
<tr>
<td><code class="cba-inline-code">time_features</code></td>
<td>TF0, TF1</td>
<td>off / sin/cos + weekend</td>
</tr>
<tr>
<td><code class="cba-inline-code">target_scaling</code></td>
<td>YS0, YS1</td>
<td>off / train_only_standardization</td>
</tr>
<tr>
<td><code class="cba-inline-code">lookbacks</code></td>
<td>L36, L72, L144</td>
<td>36 / 72 / 144 steps</td>
</tr>
<tr>
<td><code class="cba-inline-code">pooling</code></td>
<td>P0, P1</td>
<td>last_step / mean</td>
</tr>
<tr class="cba-anomaly-row">
<td><code class="cba-inline-code">activation</code></td>
<td>A0, A1</td>
<td>relu / gelu</td>
</tr>
<tr>
<td><code class="cba-inline-code">batch_size</code></td>
<td>B32, B64</td>
<td>32 / 64</td>
</tr>
<tr>
<td><code class="cba-inline-code">learning_rate</code></td>
<td>LR1, LR2, LR3</td>
<td>1e-4 / 3e-4 / 1e-3</td>
</tr>
<tr>
<td><code class="cba-inline-code">weight_decay</code></td>
<td>WD0, WD1, WD2, WD3</td>
<td>0 / 1e-4 / 1e-3 / 1e-2</td>
</tr>
<tr>
<td><code class="cba-inline-code">dropout</code></td>
<td>DR01, DR02, DR03</td>
<td>0.1 / 0.2 / 0.3</td>
</tr>
<tr>
<td><code class="cba-inline-code">d_model</code></td>
<td>D32, D64</td>
<td>32 / 64</td>
</tr>
<tr>
<td><code class="cba-inline-code">heads</code></td>
<td>H2, H4</td>
<td>2 / 4</td>
</tr>
<tr>
<td><code class="cba-inline-code">layers</code></td>
<td>N1, N2</td>
<td>1 / 2</td>
</tr>
<tr>
<td><code class="cba-inline-code">ffn</code></td>
<td>F64, F128, F256</td>
<td>64 / 128 / 256</td>
</tr>
<tr>
<td><code class="cba-inline-code">loss</code></td>
<td>L0, L1</td>
<td>mse / huber_delta_1</td>
</tr>
<tr>
<td><code class="cba-inline-code">epoch_cap</code></td>
<td>E50, E100</td>
<td>50 / 100</td>
</tr>
<tr>
<td><code class="cba-inline-code">gradient_clip</code></td>
<td>GC0, GC1</td>
<td>off / 1.0</td>
</tr>
<tr>
<td><code class="cba-inline-code">revin</code></td>
<td>RN0, RN1</td>
<td>off / on</td>
</tr>
<tr>
<td><code class="cba-inline-code">boundary</code></td>
<td>WB0, WB1</td>
<td>context_carry_over / drop_boundary</td>
</tr>
</tbody>
</table></div>
<h3 class="cba-heading cba-h3" id="54-contract-api--srccourseworkcontractscourseworkpy-">5.4. Contract API (<a class="cba-link cba-path-link" href="../src/course_work/contracts/coursework.py"><code class="cba-inline-code">src/course_work/contracts/coursework.py</code></a>)</h3>
<ul>
<li><code class="cba-inline-code">EXPECTED_OPTION_IDS</code> — set of valid IDs per group</li>
<li><code class="cba-inline-code">load_coursework_contract(path=None)</code> → dict</li>
<li><code class="cba-inline-code">validate_coursework_contract(contract)</code> → <code class="cba-inline-code">tuple[str, ...]</code> errors</li>
<li><code class="cba-inline-code">coursework_contract_fingerprint(contract)</code> → SHA-256 hex</li>
<li><code class="cba-inline-code">materialize_phase_0(project_root=None)</code> — Phase 0 entry point</li>
</ul>
<h3 class="cba-heading cba-h3" id="55-baseline-transformer-reference-config">5.5. Baseline Transformer Reference Config</h3>
<pre class="cba-code"><code class="language-json">{
  "feature_set": "FS1",
  "time_feature": "TF1",
  "target_scaling": "YS1",
  "revin": "RN0",
  "lookback": 144,
  "d_model": 64,
  "heads": 4,
  "layers": 2,
  "ffn": 128,
  "dropout": 0.1,
  "activation": "gelu",
  "pooling": "last_step",
  "batch_size": 64,
  "learning_rate": 3e-4,
  "weight_decay": 1e-4,
  "loss": "mse",
  "epochs": 50,
  "patience": 10,
  "gradient_clip": 1.0,
  "seed": 42
}
</code></pre>
<hr class="cba-rule"/>
</section><section class="cba-section cba-test-card"><h2 class="cba-heading cba-h2" id="6-tests-layer"><span class="cba-section-pill">TESTING</span>6. Tests Layer</h2>
<h3 class="cba-heading cba-h3" id="61-test-count--distribution">6.1. Test Count &amp; Distribution</h3>
<div class="cba-table-wrap"><table class="cba-table">
<thead>
<tr>
<th>Category</th>
<th>Count</th>
<th>Mục đích</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Unit</strong></td>
<td>~97</td>
<td>Per-module, per-phase</td>
</tr>
<tr>
<td><strong>Integration</strong></td>
<td>9</td>
<td>Multi-phase chains</td>
</tr>
<tr>
<td><strong>Contracts</strong></td>
<td>2</td>
<td>Option registry, execution policy</td>
</tr>
<tr>
<td><strong>Tổng</strong></td>
<td><strong>108 files</strong></td>
<td><strong>~30,500 LOC</strong></td>
</tr>
</tbody>
</table></div>
<h3 class="cba-heading cba-h3" id="62-unit-tests-pattern">6.2. Unit Tests Pattern</h3>
<ul>
<li>Mỗi sweep có 1 unit test (vd: <a class="cba-link cba-path-link" href="../tests/unit/test_d_model.py"><code class="cba-inline-code">test_d_model.py</code></a>, <a class="cba-link cba-path-link" href="../tests/unit/test_dropout.py"><code class="cba-inline-code">test_dropout.py</code></a>)</li>
<li>Mỗi phase từ 43-59 có 1+ test files</li>
<li><a class="cba-link cba-path-link" href="../tests/unit/test_phase47_infrastructure.py"><code class="cba-inline-code">test_phase47_infrastructure.py</code></a> (1,080 LOC) — strictest invariants</li>
<li><a class="cba-link cba-path-link" href="../tests/unit/test_phase43_corrective_implementation.py"><code class="cba-inline-code">test_phase43_corrective_implementation.py</code></a> (1,763 LOC) — lớn nhất</li>
<li>5 RevIN-specific tests</li>
</ul>
<h3 class="cba-heading cba-h3" id="63-integration-tests-pattern">6.3. Integration Tests Pattern</h3>
<ul>
<li><a class="cba-link cba-path-link" href="../tests/integration/test_phase_0_to_5_chain.py"><code class="cba-inline-code">test_phase_0_to_5_chain.py</code></a> (318 LOC) — phase 0→5 end-to-end</li>
<li><a class="cba-link cba-path-link" href="../tests/integration/test_phase_0_to_8_presentation.py"><code class="cba-inline-code">test_phase_0_to_8_presentation.py</code></a> — phase 0→8</li>
<li><a class="cba-link cba-path-link" href="../tests/integration/test_notebook_boundary.py"><code class="cba-inline-code">test_notebook_boundary.py</code></a> (515 LOC) — notebook kernel/sandbox</li>
<li><a class="cba-link cba-path-link" href="../tests/integration/test_final_dev_dataset.py"><code class="cba-inline-code">test_final_dev_dataset.py</code></a> (564 LOC) — FINAL_DEV region</li>
<li><code class="cba-inline-code">test_phase46_*.py</code> (3 files) — pretrain gates</li>
<li><a class="cba-link cba-path-link" href="../tests/unit/test_phase47_infrastructure.py"><code class="cba-inline-code">test_phase47_infrastructure.py</code></a> — test firewall enforcement</li>
<li><a class="cba-link cba-path-link" href="../tests/integration/test_mape_addendum.py"><code class="cba-inline-code">test_mape_addendum.py</code></a> — MAPE addendum</li>
<li><a class="cba-link cba-path-link" href="../tests/integration/dry_run_final_dev.py"><code class="cba-inline-code">dry_run_final_dev.py</code></a> — no-train run</li>
</ul>
<h3 class="cba-heading cba-h3" id="64-contract-tests">6.4. Contract Tests</h3>
<ul>
<li><a class="cba-link cba-path-link" href="../tests/contracts/test_coursework_contract.py"><code class="cba-inline-code">test_coursework_contract.py</code></a> — validate option registry</li>
<li><a class="cba-link cba-path-link" href="../tests/contracts/test_selective_execution_policy.py"><code class="cba-inline-code">test_selective_execution_policy.py</code></a> (283 LOC) — phase resume rules</li>
</ul>
<h3 class="cba-heading cba-h3" id="65-conftestpy-setup">6.5. <a class="cba-link cba-path-link" href="../tests/conftest.py"><code class="cba-inline-code">conftest.py</code></a> Setup</h3>
<pre class="cba-code"><code class="language-python"># tests/conftest.py adds src/ to sys.path,
# sets os.environ["COURSE_WORK_ROOT"],
# calls os.chdir() so get_project_root() works.
</code></pre>
<hr class="cba-rule"/>
</section><section class="cba-section cba-artifact-card"><h2 class="cba-heading cba-h2" id="7-artifacts-layer"><span class="cba-section-pill">ARTIFACTS</span>7. Artifacts Layer</h2>
<h3 class="cba-heading cba-h3" id="71-phase-artifacts-49-top-level-directories">7.1. Phase Artifacts (49 top-level directories)</h3>
<div class="cba-table-wrap"><table class="cba-table">
<thead>
<tr>
<th>#</th>
<th>Directory</th>
<th>Phase</th>
<th>Key Files</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><code class="cba-inline-code">contracts/</code></td>
<td>0</td>
<td><a class="cba-link cba-path-link" href="../artifacts/contracts/coursework_contract.json"><code class="cba-inline-code">coursework_contract.json</code></a>, <a class="cba-link cba-path-link" href="../artifacts/contracts/phase_0_signoff.json"><code class="cba-inline-code">phase_0_signoff.json</code></a></td>
</tr>
<tr>
<td>2</td>
<td><code class="cba-inline-code">environment/</code></td>
<td>1</td>
<td><a class="cba-link cba-path-link" href="../artifacts/environment/environment_report.json"><code class="cba-inline-code">environment_report.json</code></a>, <a class="cba-link cba-path-link" href="../artifacts/environment/requirements_freeze.txt"><code class="cba-inline-code">requirements_freeze.txt</code></a></td>
</tr>
<tr>
<td>3</td>
<td><code class="cba-inline-code">acquisition/</code></td>
<td>2</td>
<td><a class="cba-link cba-path-link" href="../artifacts/acquisition/acquisition_log.json"><code class="cba-inline-code">acquisition_log.json</code></a>, <a class="cba-link cba-path-link" href="../artifacts/acquisition/phase_2_signoff.json"><code class="cba-inline-code">phase_2_signoff.json</code></a></td>
</tr>
<tr>
<td>4</td>
<td><code class="cba-inline-code">schema/</code></td>
<td>3</td>
<td><a class="cba-link cba-path-link" href="../artifacts/schema/schema_manifest.json"><code class="cba-inline-code">schema_manifest.json</code></a></td>
</tr>
<tr>
<td>5</td>
<td><code class="cba-inline-code">temporal/</code></td>
<td>4</td>
<td><a class="cba-link cba-path-link" href="../artifacts/temporal/temporal_manifest.json"><code class="cba-inline-code">temporal_manifest.json</code></a></td>
</tr>
<tr>
<td>6</td>
<td><code class="cba-inline-code">eda/</code></td>
<td>5</td>
<td><a class="cba-link cba-path-link" href="../artifacts/eda/eda_manifest.json"><code class="cba-inline-code">eda_manifest.json</code></a>, <code class="cba-inline-code">figures/</code>, <code class="cba-inline-code">tables/</code></td>
</tr>
<tr>
<td>7</td>
<td><code class="cba-inline-code">features/</code></td>
<td>6</td>
<td><a class="cba-link cba-path-link" href="../artifacts/features/feature_engineering_manifest.json"><code class="cba-inline-code">feature_engineering_manifest.json</code></a></td>
</tr>
<tr>
<td>8</td>
<td><code class="cba-inline-code">feature_sets/</code></td>
<td>7</td>
<td><a class="cba-link cba-path-link" href="../artifacts/feature_sets/feature_set_manifest.json"><code class="cba-inline-code">feature_set_manifest.json</code></a></td>
</tr>
<tr>
<td>9</td>
<td><code class="cba-inline-code">splits/</code></td>
<td>8</td>
<td><a class="cba-link cba-path-link" href="../artifacts/splits/split_manifest.json"><code class="cba-inline-code">split_manifest.json</code></a>, <a class="cba-link cba-path-link" href="../artifacts/splits/split_membership.csv"><code class="cba-inline-code">split_membership.csv</code></a></td>
</tr>
<tr>
<td>10</td>
<td><code class="cba-inline-code">scaling/</code></td>
<td>9</td>
<td><a class="cba-link cba-path-link" href="../artifacts/scaling/scaling_manifest.json"><code class="cba-inline-code">scaling_manifest.json</code></a>, <a class="cba-link cba-path-link" href="../artifacts/scaling/scaler_registry.json"><code class="cba-inline-code">scaler_registry.json</code></a></td>
</tr>
<tr>
<td>11</td>
<td><code class="cba-inline-code">scalers/</code></td>
<td>9</td>
<td>(joblib files - empty, see anomaly)</td>
</tr>
<tr>
<td>12</td>
<td><code class="cba-inline-code">windows/</code></td>
<td>10</td>
<td><a class="cba-link cba-path-link" href="../artifacts/windows/window_manifest.json"><code class="cba-inline-code">window_manifest.json</code></a>, <a class="cba-link cba-path-link" href="../artifacts/windows/window_index.csv"><code class="cba-inline-code">window_index.csv</code></a></td>
</tr>
<tr>
<td>13</td>
<td><code class="cba-inline-code">dataloaders/</code></td>
<td>11</td>
<td><a class="cba-link cba-path-link" href="../artifacts/dataloaders/dataloader_manifest.json"><code class="cba-inline-code">dataloader_manifest.json</code></a></td>
</tr>
<tr>
<td>14</td>
<td><code class="cba-inline-code">metrics/</code></td>
<td>12</td>
<td><a class="cba-link cba-path-link" href="../artifacts/metrics/metric_manifest.json"><code class="cba-inline-code">metric_manifest.json</code></a></td>
</tr>
<tr>
<td>15</td>
<td><code class="cba-inline-code">experiments/</code></td>
<td>13</td>
<td><code class="cba-inline-code">experiment_registry.jsonl</code>, <code class="cba-inline-code">registry_manifest.json</code></td>
</tr>
<tr>
<td>16</td>
<td><code class="cba-inline-code">baselines/persistence/</code></td>
<td>14</td>
<td><code class="cba-inline-code">persistence_manifest.json</code></td>
</tr>
<tr>
<td>17</td>
<td><code class="cba-inline-code">models/lstm/</code></td>
<td>15</td>
<td><code class="cba-inline-code">lstm_model_manifest.json</code></td>
</tr>
<tr>
<td>18</td>
<td><code class="cba-inline-code">models/transformer/</code></td>
<td>16</td>
<td><code class="cba-inline-code">transformer_model_manifest.json</code></td>
</tr>
<tr>
<td>19</td>
<td><code class="cba-inline-code">attention_verification/</code></td>
<td>17</td>
<td><code class="cba-inline-code">attention_verification_manifest.json</code></td>
</tr>
<tr>
<td>20</td>
<td><code class="cba-inline-code">forward_sanity/</code></td>
<td>18</td>
<td><code class="cba-inline-code">forward_sanity_manifest.json</code></td>
</tr>
<tr>
<td>21</td>
<td><code class="cba-inline-code">training_engine/</code></td>
<td>19</td>
<td><code class="cba-inline-code">training_engine_manifest.json</code></td>
</tr>
<tr>
<td>22</td>
<td><code class="cba-inline-code">lstm_baseline/</code></td>
<td>20</td>
<td><a class="cba-link cba-path-link" href="../artifacts/lstm_baseline/lstm_baseline_run_summary.csv"><code class="cba-inline-code">lstm_baseline_run_summary.csv</code></a></td>
</tr>
<tr>
<td>23</td>
<td><code class="cba-inline-code">transformer_b0/</code></td>
<td>21</td>
<td><a class="cba-link cba-path-link" href="../artifacts/transformer_b0/transformer_b0_run_summary.csv"><code class="cba-inline-code">transformer_b0_run_summary.csv</code></a></td>
</tr>
<tr>
<td>24</td>
<td><code class="cba-inline-code">learning_diagnostics/</code></td>
<td>22</td>
<td><code class="cba-inline-code">learning_diagnostics_manifest.json</code></td>
</tr>
<tr>
<td>25</td>
<td><code class="cba-inline-code">sweeps/s1..s8/</code></td>
<td>23-30</td>
<td>per-sweep manifests, winners</td>
</tr>
<tr>
<td>26</td>
<td><code class="cba-inline-code">sweeps/S9..S19/</code></td>
<td>31-41</td>
<td>per-sweep manifests, winners</td>
</tr>
<tr>
<td>27</td>
<td><code class="cba-inline-code">candidate_synthesis/</code></td>
<td>42</td>
<td><a class="cba-link cba-path-link" href="../artifacts/candidate_synthesis/candidate_synthesis_manifest.json"><code class="cba-inline-code">candidate_synthesis_manifest.json</code></a></td>
</tr>
<tr>
<td>28</td>
<td><code class="cba-inline-code">lstm_tuning/</code></td>
<td>43</td>
<td><code class="cba-inline-code">lstm_tuning_manifest.json</code></td>
</tr>
<tr>
<td>29</td>
<td><code class="cba-inline-code">rolling_origin/</code></td>
<td>44</td>
<td>39 O44.* artifacts</td>
</tr>
<tr>
<td>30</td>
<td><code class="cba-inline-code">final_model_lock/</code></td>
<td>45</td>
<td>38 O45.* artifacts</td>
</tr>
<tr>
<td>31</td>
<td><code class="cba-inline-code">final_dev_region/</code></td>
<td>46</td>
<td>FINAL_DEV region manifest</td>
</tr>
<tr>
<td>32</td>
<td><code class="cba-inline-code">three_seed_final_runs/</code></td>
<td>46</td>
<td>Official checkpoints</td>
</tr>
<tr>
<td>33</td>
<td><code class="cba-inline-code">final_test/</code></td>
<td>47</td>
<td><code class="cba-inline-code">final_test_evaluation_contract.json</code></td>
</tr>
<tr>
<td>34</td>
<td><code class="cba-inline-code">prediction_analysis/</code></td>
<td>48</td>
<td>per-phase dashboard</td>
</tr>
<tr>
<td>35</td>
<td><code class="cba-inline-code">residual_analysis/</code></td>
<td>49</td>
<td>per-phase dashboard</td>
</tr>
<tr>
<td>36</td>
<td><code class="cba-inline-code">error_by_regime/</code></td>
<td>50</td>
<td>per-phase dashboard</td>
</tr>
<tr>
<td>37</td>
<td><code class="cba-inline-code">worst_error_analysis/</code></td>
<td>51</td>
<td>per-phase dashboard</td>
</tr>
<tr>
<td>38</td>
<td><code class="cba-inline-code">attention_extraction/</code></td>
<td>52</td>
<td>per-phase dashboard</td>
</tr>
<tr>
<td>39</td>
<td><code class="cba-inline-code">attention_heatmaps/</code></td>
<td>53</td>
<td>per-phase dashboard</td>
</tr>
<tr>
<td>40</td>
<td><code class="cba-inline-code">last_query_attention/</code></td>
<td>54</td>
<td>per-phase dashboard</td>
</tr>
<tr>
<td>41</td>
<td><code class="cba-inline-code">head_comparison/</code></td>
<td>55</td>
<td>per-phase dashboard</td>
</tr>
<tr>
<td>42</td>
<td><code class="cba-inline-code">error_conditioned_attention/</code></td>
<td>56</td>
<td>per-phase dashboard</td>
</tr>
<tr>
<td>43</td>
<td><code class="cba-inline-code">seed_stability_attention/</code></td>
<td>57</td>
<td>per-phase dashboard</td>
</tr>
<tr>
<td>44</td>
<td><code class="cba-inline-code">final_tables/</code></td>
<td>58</td>
<td>per-phase dashboard</td>
</tr>
<tr>
<td>45</td>
<td><code class="cba-inline-code">final_conclusions/</code></td>
<td>59</td>
<td>per-phase dashboard</td>
</tr>
<tr>
<td>46</td>
<td><code class="cba-inline-code">runs/</code></td>
<td>various</td>
<td>~25 RUN_* directories</td>
</tr>
<tr>
<td>-</td>
<td><code class="cba-inline-code">_history/</code></td>
<td>recovery</td>
<td>timestamped snapshots</td>
</tr>
</tbody>
</table></div>
<h3 class="cba-heading cba-h3" id="72-universal-artifact-schema">7.2. Universal Artifact Schema</h3>
<p>Mỗi phase directory có các file pattern:</p>
<ul>
<li><code class="cba-inline-code">phase_N_signoff.json</code> — gate với status PASS/FAIL</li>
<li><code class="cba-inline-code">phase_N_manifest.json</code> — input/output checksums</li>
<li><code class="cba-inline-code">phase_N_contract.json</code> — config lock</li>
<li><code class="cba-inline-code">phase_N_report.md</code> — human-readable summary</li>
<li><code class="cba-inline-code">phase_N_discrepancies.json</code> — issues tracker</li>
<li><code class="cba-inline-code">figures/</code>, <code class="cba-inline-code">tables/</code> — visualization</li>
</ul>
<h3 class="cba-heading cba-h3" id="73-history-mechanism">7.3. <code class="cba-inline-code">_history/</code> Mechanism</h3>
<ul>
<li><code class="cba-inline-code">utils.environment.recover_environment_revision()</code> move existing artifacts</li>
<li>Snapshot pattern: <code class="cba-inline-code">recover_X()</code> → move to <code class="cba-inline-code">_history/&lt;UTC_stamp&gt;/</code></li>
<li>Cho phép rollback mà không mất dữ liệu</li>
</ul>
<hr class="cba-rule"/>
</section><section class="cba-section cba-data-card"><h2 class="cba-heading cba-h2" id="8-data-layer"><span class="cba-section-pill">DATA</span>8. Data Layer</h2>
<h3 class="cba-heading cba-h3" id="81-canonical-data-location">8.1. Canonical Data Location</h3>
<p><strong>Hiện tại:</strong> <a class="cba-link cba-path-link" href="../link/raw_data"><code class="cba-inline-code">link/raw_data/</code></a>, <code class="cba-inline-code">link/interim/</code></p>
<div class="cba-table-wrap"><table class="cba-table">
<thead>
<tr>
<th>Path</th>
<th>Nội dung</th>
</tr>
</thead>
<tbody>
<tr>
<td><code class="cba-inline-code">link/raw_data/energydata_complete.csv</code></td>
<td>UCI CSV, ~19,735 rows × 29 cols</td>
</tr>
<tr>
<td><a class="cba-link cba-path-link" href="../link/raw_data/dataset_manifest.json"><code class="cba-inline-code">link/raw_data/dataset_manifest.json</code></a></td>
<td>SHA-256 + metadata</td>
</tr>
<tr>
<td><a class="cba-link cba-path-link" href="../link/raw_data/source_metadata.json"><code class="cba-inline-code">link/raw_data/source_metadata.json</code></a></td>
<td>Source provenance</td>
</tr>
<tr>
<td><code class="cba-inline-code">link/raw_data/variable_metadata.csv</code></td>
<td>29 variable dictionary</td>
</tr>
<tr>
<td><a class="cba-link cba-path-link" href="../link/raw_data/checksums.sha256"><code class="cba-inline-code">link/raw_data/checksums.sha256</code></a></td>
<td>Integrity checksums</td>
</tr>
<tr>
<td><a class="cba-link cba-path-link" href="../link/raw_data/README_SOURCE.md"><code class="cba-inline-code">link/raw_data/README_SOURCE.md</code></a></td>
<td>Source documentation</td>
</tr>
<tr>
<td><code class="cba-inline-code">link/raw_data/source/appliances_energy_prediction.zip</code></td>
<td>Protected archive</td>
</tr>
<tr>
<td><code class="cba-inline-code">link/interim/uci_appliances_energy_prediction/energydata_feature_engineered_v1.csv</code></td>
<td>Feature-engineered output</td>
</tr>
</tbody>
</table></div>
<h3 class="cba-heading cba-h3" id="82--data-path-anomaly">8.2. ⚠️ Data Path Anomaly</h3>
<ul>
<li>Code references: <code class="cba-inline-code">data/raw_data/...</code></li>
<li>Physical location: <code class="cba-inline-code">link/raw_data/...</code></li>
<li>Có thể <code class="cba-inline-code">data/</code> là symlink → <a class="cba-link cba-path-link" href="../link"><code class="cba-inline-code">link/</code></a> hoặc runtime remap. Cần verify.</li>
</ul>
<h3 class="cba-heading cba-h3" id="83-missing-directories">8.3. Missing Directories</h3>
<ul>
<li><code class="cba-inline-code">data_after_processing/</code> — KHÔNG tồn tại</li>
<li><code class="cba-inline-code">data_after_split/</code> — KHÔNG tồn tại</li>
<li>(Có thể do cleanup trong lần refactor gần đây)</li>
</ul>
<hr class="cba-rule"/>
</section><section class="cba-section cba-notebook-card"><h2 class="cba-heading cba-h2" id="9-notebook-layer"><span class="cba-section-pill">NOTEBOOK</span>9. Notebook Layer</h2>
<h3 class="cba-heading cba-h3" id="91-structure">9.1. Structure</h3>
<ul>
<li><strong>File:</strong> <a class="cba-link cba-notebook-link" href="../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Markdown links do not reliably deep-link to an exact notebook cell."><code class="cba-inline-code">notebook_course_work/CourseWork.ipynb</code></a></li>
<li><strong>Cells:</strong> 153</li>
<li><strong>Vai trò:</strong> orchestration và presentation; scientific computation nằm trong <a class="cba-link cba-path-link" href="../src/course_work"><code class="cba-inline-code">src/course_work/</code></a>.</li>
<li><strong>Canonical mapping:</strong> <a class="cba-link" href="link&amp;discussion_to_result/NOTEBOOK_CELLS_WALKTHROUGH.md"><code class="cba-inline-code">NOTEBOOK_CELLS_WALKTHROUGH.md</code></a>.</li>
</ul>
<h3 class="cba-heading cba-h3" id="92-cell-to-phase-mapping-hiện-tại">9.2. Cell-to-Phase Mapping hiện tại</h3>
<div class="cba-table-wrap"><table class="cba-table">
<thead>
<tr>
<th>Cells</th>
<th>Phase</th>
<th>Renderer / entry point</th>
<th>Lineage</th>
</tr>
</thead>
<tbody>
<tr>
<td>0–3</td>
<td>Intro</td>
<td>Notebook setup</td>
<td>Presentation</td>
</tr>
<tr>
<td>4–74</td>
<td>1–21</td>
<td><a class="cba-link" href="../src/course_work/reporting/phase_summary.py"><code class="cba-inline-code">phase_summary.py</code></a></td>
<td>V1</td>
</tr>
<tr>
<td>75–96</td>
<td>22–32</td>
<td><a class="cba-link" href="../src/course_work/reporting/frozen_evidence.py"><code class="cba-inline-code">frozen_evidence.py</code></a></td>
<td>V1</td>
</tr>
<tr>
<td>97–98</td>
<td>33</td>
<td><code class="cba-inline-code">render_phase_33_transformer_configuration</code></td>
<td>V1</td>
</tr>
<tr>
<td>99–114</td>
<td>34–41</td>
<td><a class="cba-link" href="../src/course_work/reporting/frozen_evidence.py"><code class="cba-inline-code">frozen_evidence.py</code></a></td>
<td><code class="cba-inline-code cba-code-blue">V1 RECOVERED</code></td>
</tr>
<tr>
<td>115–124</td>
<td>42–46</td>
<td>verified historical renderer</td>
<td><code class="cba-inline-code cba-code-blue">V1 HISTORICAL</code></td>
</tr>
<tr class="cba-v2-row">
<td>125–134</td>
<td>47–51</td>
<td><a class="cba-link" href="../src/course_work/reporting/results_rebuild.py"><code class="cba-inline-code">results_rebuild.py</code></a></td>
<td><code class="cba-inline-code cba-code-green">V2 FINAL</code></td>
</tr>
<tr>
<td>135–146</td>
<td>52–57</td>
<td><a class="cba-link" href="../src/course_work/reporting/results_rebuild.py"><code class="cba-inline-code">results_rebuild.py</code></a></td>
<td><code class="cba-inline-code cba-code-blue">V1 HISTORICAL ATTENTION</code></td>
</tr>
<tr class="cba-v2-row">
<td>147–150</td>
<td>58–59</td>
<td><a class="cba-link" href="../src/course_work/reporting/results_rebuild.py"><code class="cba-inline-code">results_rebuild.py</code></a></td>
<td><code class="cba-inline-code cba-code-green">V2 FINAL</code></td>
</tr>
<tr class="cba-v2-row">
<td>151–152</td>
<td>V2 overview</td>
<td><code class="cba-inline-code">render_verified_v2_results</code></td>
<td><code class="cba-inline-code cba-code-green">V2 FINAL</code></td>
</tr>
</tbody>
</table></div>
<h3 class="cba-heading cba-h3" id="93-notebook-lineage">9.3. Notebook lineage</h3>
<p>Historical V1 artifacts vẫn được giữ làm provenance. Renderer hiện tại không đổi nhãn artifact V1 thành V2. Phase 52–57 không được dùng để kết luận attention behavior của final V2 vì V2 Test attention tensors không được tạo.</p>
<hr class="cba-rule"/>
</section><section class="cba-section cba-docs-card"><h2 class="cba-heading cba-h2" id="10-documentation-layer"><span class="cba-section-pill">DOCS</span>10. Documentation Layer</h2>
<h3 class="cba-heading cba-h3" id="101-docs-structure">10.1. <a class="cba-link cba-path-link" href="../docs"><code class="cba-inline-code">docs/</code></a> Structure</h3>
<pre class="cba-code"><code>docs/
├── analysis_error/                 # 79 issue analysis files
├── code_base_audit.md              # File này
├── current_flow/
│   ├── CURRENT_FLOW_SUMMARY.md            # 410 lines
│   ├── PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md  # 523 lines
│   └── PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md # 505 lines
├── link&amp;discussion_to_result/
│   └── NOTEBOOK_CELLS_WALKTHROUGH.md      # 1,094 lines
├── plan/
│   ├── plan_before_process/         # Pre-process plans (~30 files)
│   ├── plan_detail_for_each_phase/
│   ├── plan_overview/
│   └── plan_to_refactor&amp;fix/        # Recent refactor plans
├── rule_base/
│   ├── architecture_rule.md
│   └── rule_code.md
└── save_log_in_processing/          # 112 JSON processing logs
</code></pre>
<h3 class="cba-heading cba-h3" id="102-doc-usage-guide">10.2. Doc Usage Guide</h3>
<div class="cba-table-wrap"><table class="cba-table">
<thead>
<tr>
<th>Doc</th>
<th>Khi nào đọc</th>
</tr>
</thead>
<tbody>
<tr>
<td><a class="cba-link cba-path-link" href="../docs/code_base_audit.md"><code class="cba-inline-code">code_base_audit.md</code></a></td>
<td>Hiểu tổng quan toàn project</td>
</tr>
<tr>
<td><a class="cba-link cba-path-link" href="../docs/current_flow/CURRENT_FLOW_SUMMARY.md"><code class="cba-inline-code">CURRENT_FLOW_SUMMARY.md</code></a></td>
<td>Quick reference về kiến trúc</td>
</tr>
<tr>
<td><a class="cba-link cba-path-link" href="../docs/current_flow/PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md"><code class="cba-inline-code">PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md</code></a></td>
<td>Chi tiết foundation + modeling</td>
</tr>
<tr>
<td><a class="cba-link cba-path-link" href="../docs/current_flow/PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md"><code class="cba-inline-code">PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md</code></a></td>
<td>Chi tiết sweeps + final + analysis</td>
</tr>
<tr>
<td><a class="cba-link cba-path-link" href="../docs/link&discussion_to_result/NOTEBOOK_CELLS_WALKTHROUGH.md"><code class="cba-inline-code">NOTEBOOK_CELLS_WALKTHROUGH.md</code></a></td>
<td>Đọc notebook output</td>
</tr>
<tr>
<td><code class="cba-inline-code">analysis_error/&lt;file&gt;.md</code></td>
<td>Khi gặp bug, tìm issue đã biết</td>
</tr>
<tr>
<td><a class="cba-link cba-path-link" href="../docs/rule_base/architecture_rule.md"><code class="cba-inline-code">rule_base/architecture_rule.md</code></a></td>
<td>Conventions</td>
</tr>
<tr>
<td><a class="cba-link cba-path-link" href="../docs/rule_base/rule_code.md"><code class="cba-inline-code">rule_base/rule_code.md</code></a></td>
<td>Coding style</td>
</tr>
</tbody>
</table></div>
<hr class="cba-rule"/>
</section><section class="cba-section cba-phase-card"><h2 class="cba-heading cba-h2" id="11-phase-mapping-chi-tiết-0-59"><span class="cba-section-pill">PHASE MAP</span>11. Phase Mapping Chi Tiết (0-59)</h2>
<h3 class="cba-heading cba-h3" id="111-foundation-0-10">11.1. Foundation (0-10)</h3>
<div class="cba-table-wrap"><table class="cba-table">
<thead>
<tr>
<th>Phase</th>
<th>Primary Module</th>
<th>Script</th>
<th>Purpose</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td><code class="cba-inline-code">contracts.coursework</code></td>
<td>—</td>
<td>Contract materialization</td>
</tr>
<tr>
<td>1</td>
<td><code class="cba-inline-code">utils.environment</code></td>
<td>—</td>
<td>Env report, freeze, smoke test</td>
</tr>
<tr>
<td>2</td>
<td><code class="cba-inline-code">data.acquisition</code></td>
<td>—</td>
<td>UCI ZIP + SHA-256</td>
</tr>
<tr>
<td>3</td>
<td><code class="cba-inline-code">data.schema</code></td>
<td>—</td>
<td>Schema audit</td>
</tr>
<tr>
<td>4</td>
<td><code class="cba-inline-code">data.temporal</code></td>
<td>—</td>
<td>Temporal integrity</td>
</tr>
<tr>
<td>5</td>
<td><code class="cba-inline-code">data.splitting</code> (artifact: splits)</td>
<td>—</td>
<td>Chronological split</td>
</tr>
<tr>
<td>6</td>
<td><code class="cba-inline-code">data.features</code> (artifact: features)</td>
<td>—</td>
<td>Feature engineering</td>
</tr>
<tr>
<td>7</td>
<td><code class="cba-inline-code">data.feature_sets</code></td>
<td>—</td>
<td>Feature-set variants</td>
</tr>
<tr>
<td>8</td>
<td><code class="cba-inline-code">data.scaling</code> (artifact: scaling)</td>
<td>—</td>
<td>Train-only scaling</td>
</tr>
<tr>
<td>9</td>
<td><code class="cba-inline-code">data.windows</code> (artifact: windows)</td>
<td>—</td>
<td>Window builder</td>
</tr>
<tr>
<td>10</td>
<td><code class="cba-inline-code">data.datasets</code> (artifact: dataloaders)</td>
<td>—</td>
<td>DataLoaders</td>
</tr>
<tr>
<td>11</td>
<td><code class="cba-inline-code">data.datasets</code> (artifact: dataloaders)</td>
<td>—</td>
<td>DataLoaders</td>
</tr>
</tbody>
</table></div>
<h3 class="cba-heading cba-h3" id="112-modeling-12-22">11.2. Modeling (12-22)</h3>
<div class="cba-table-wrap"><table class="cba-table">
<thead>
<tr>
<th>Phase</th>
<th>Primary Module</th>
<th>Purpose</th>
</tr>
</thead>
<tbody>
<tr>
<td>12</td>
<td><code class="cba-inline-code">evaluation.metrics</code></td>
<td>Shared metrics (MAE/RMSE/R²/MAPE)</td>
</tr>
<tr>
<td>13</td>
<td><code class="cba-inline-code">experiments.registry</code></td>
<td>Experiment registry</td>
</tr>
<tr>
<td>14</td>
<td><code class="cba-inline-code">baselines.persistence</code></td>
<td>Persistence baseline</td>
</tr>
<tr>
<td>15</td>
<td><code class="cba-inline-code">models.lstm_regressor</code></td>
<td>LSTM architecture</td>
</tr>
<tr>
<td>16</td>
<td><code class="cba-inline-code">models.transformer_regressor</code></td>
<td>Transformer architecture</td>
</tr>
<tr>
<td>17</td>
<td><code class="cba-inline-code">attention.verification</code></td>
<td>Attention contract audit</td>
</tr>
<tr>
<td>18</td>
<td><code class="cba-inline-code">sanity.forward_sanity</code></td>
<td>Forward pass tests</td>
</tr>
<tr>
<td>19</td>
<td><code class="cba-inline-code">training.engine</code></td>
<td>Generic training engine</td>
</tr>
<tr>
<td>20</td>
<td><code class="cba-inline-code">baselines.lstm_baseline</code></td>
<td>LSTM_B0 training run</td>
</tr>
<tr>
<td>21</td>
<td><code class="cba-inline-code">baselines.transformer_b0</code></td>
<td>Transformer_B0 training run</td>
</tr>
<tr>
<td>22</td>
<td><code class="cba-inline-code">diagnostics.learning_diagnostics</code></td>
<td>Learning curve analysis</td>
</tr>
</tbody>
</table></div>
<h3 class="cba-heading cba-h3" id="113-sweeps-23-41">11.3. Sweeps (23-41)</h3>
<div class="cba-table-wrap"><table class="cba-table">
<thead>
<tr>
<th>Phase</th>
<th>Primary Module</th>
<th>Factor</th>
<th>Conditions</th>
</tr>
</thead>
<tbody>
<tr>
<td>23</td>
<td><code class="cba-inline-code">sweeps.sweep_results</code> (feature_set)</td>
<td>feature_set</td>
<td>FS0, FS1, FS2</td>
</tr>
<tr>
<td>24</td>
<td><code class="cba-inline-code">sweeps.sweep_results</code> (time_feature)</td>
<td>time_feature</td>
<td>TF0, TF1</td>
</tr>
<tr>
<td>25</td>
<td><code class="cba-inline-code">sweeps.sweep_results</code> (target_scaling)</td>
<td>target_scaling</td>
<td>YS0, YS1</td>
</tr>
<tr>
<td>26</td>
<td><code class="cba-inline-code">sweeps.sweep_results</code> (lookback)</td>
<td>lookback</td>
<td>L36, L72, L144</td>
</tr>
<tr>
<td>27</td>
<td><code class="cba-inline-code">sweeps.sweep_results</code> (pooling)</td>
<td>pooling</td>
<td>P0, P1</td>
</tr>
<tr class="cba-anomaly-row">
<td>28</td>
<td><code class="cba-inline-code">sweeps.sweep_results</code> (activation)</td>
<td>activation</td>
<td>A0, A1</td>
</tr>
<tr>
<td>29</td>
<td><code class="cba-inline-code">sweeps.sweep_results</code> (batch_size)</td>
<td>batch_size</td>
<td>B32, B64</td>
</tr>
<tr>
<td>30</td>
<td><code class="cba-inline-code">sweeps.sweep_results</code> (learning_rate)</td>
<td>learning_rate</td>
<td>LR1, LR2, LR3</td>
</tr>
<tr>
<td>31</td>
<td><code class="cba-inline-code">sweeps.weight_decay</code></td>
<td>weight_decay</td>
<td>WD0, WD1, WD2, WD3</td>
</tr>
<tr>
<td>32</td>
<td><code class="cba-inline-code">sweeps.dropout</code></td>
<td>dropout</td>
<td>DR01, DR02, DR03</td>
</tr>
<tr>
<td>33</td>
<td><code class="cba-inline-code">sweeps.d_model</code></td>
<td>d_model</td>
<td>D32, D64</td>
</tr>
<tr>
<td>34</td>
<td><code class="cba-inline-code">sweeps.heads</code></td>
<td>num_heads</td>
<td>H2, H4</td>
</tr>
<tr>
<td>35</td>
<td><code class="cba-inline-code">sweeps.layers</code></td>
<td>num_layers</td>
<td>N1, N2</td>
</tr>
<tr>
<td>36</td>
<td><code class="cba-inline-code">sweeps.ffn</code></td>
<td>ffn_dim</td>
<td>F64, F128, F256</td>
</tr>
<tr>
<td>37</td>
<td><code class="cba-inline-code">sweeps.loss</code></td>
<td>loss_name</td>
<td>L0 (MSE), L1 (Huber)</td>
</tr>
<tr>
<td>38</td>
<td><code class="cba-inline-code">sweeps.epoch_cap</code></td>
<td>max_epochs</td>
<td>E50, E100</td>
</tr>
<tr>
<td>39</td>
<td><code class="cba-inline-code">sweeps.gradient_clip</code></td>
<td>gradient_clipping</td>
<td>GC0 (off), GC1 (1.0)</td>
</tr>
<tr>
<td>40</td>
<td><code class="cba-inline-code">sweeps.revin</code> + <code class="cba-inline-code">models.revin</code></td>
<td>revin_enabled</td>
<td>RN0, RN1</td>
</tr>
<tr>
<td>41</td>
<td><code class="cba-inline-code">sweeps.boundary_protocol</code></td>
<td>boundary</td>
<td>WB0, WB1</td>
</tr>
</tbody>
</table></div>
<h3 class="cba-heading cba-h3" id="114-candidate--lock-42-47">11.4. Candidate &amp; Lock (42-47)</h3>
<div class="cba-table-wrap"><table class="cba-table">
<thead>
<tr>
<th>Phase</th>
<th>Scientific lineage</th>
<th>Primary module</th>
<th>Current purpose/presentation</th>
</tr>
</thead>
<tbody>
<tr>
<td>42</td>
<td>V1</td>
<td><code class="cba-inline-code">experiments.phase_execution</code></td>
<td>Candidate synthesis</td>
</tr>
<tr>
<td>43</td>
<td>V1</td>
<td><code class="cba-inline-code">lstm_tuning.stages</code></td>
<td>Historical LSTM tuning</td>
</tr>
<tr>
<td>44</td>
<td>V1</td>
<td><code class="cba-inline-code">rolling_origin.pipeline</code></td>
<td>Historical rolling-origin robustness</td>
</tr>
<tr>
<td>45</td>
<td>V1</td>
<td><code class="cba-inline-code">final_model_lock.candidate_lock</code></td>
<td>Historical no-train lock</td>
</tr>
<tr>
<td>46</td>
<td>V1</td>
<td><code class="cba-inline-code">scaling.final_scaling</code> + <code class="cba-inline-code">data.final_dev</code></td>
<td>Historical three-seed refit</td>
</tr>
<tr class="cba-v2-row">
<td>47</td>
<td>V2 FINAL presentation</td>
<td><a class="cba-link" href="../src/course_work/reporting/results_rebuild.py"><code class="cba-inline-code">results_rebuild.py</code></a></td>
<td>Step 17 final locked V2 policy; V1 Phase 47 remains immutable provenance</td>
</tr>
</tbody>
</table></div>
<h3 class="cba-heading cba-h3" id="115-analysis-48-59">11.5. Analysis (48-59)</h3>
<div class="cba-table-wrap"><table class="cba-table">
<thead>
<tr>
<th>Phase</th>
<th>Current lineage</th>
<th>Purpose</th>
</tr>
</thead>
<tbody>
<tr class="cba-v2-row">
<td>48</td>
<td>V2 FINAL</td>
<td>Prediction distribution/change/peak/Persistence</td>
</tr>
<tr class="cba-v2-row">
<td>49</td>
<td>V2 FINAL</td>
<td>Residual/bias/error distribution</td>
</tr>
<tr class="cba-v2-row">
<td>50</td>
<td>V2 FINAL</td>
<td>Error-by-regime on verified matched population</td>
</tr>
<tr class="cba-v2-row">
<td>51</td>
<td>V2 FINAL</td>
<td>Worst-error ranking/case analysis</td>
</tr>
<tr>
<td>52–57</td>
<td>V1 HISTORICAL ATTENTION</td>
<td>Historical attention analyses; not final V2 evidence</td>
</tr>
<tr class="cba-v2-row">
<td>58</td>
<td>V2 FINAL</td>
<td>Final V2 summary</td>
</tr>
<tr class="cba-v2-row">
<td>59</td>
<td>V2 FINAL</td>
<td>Final V2 conclusions/closure</td>
</tr>
</tbody>
</table></div>
<hr class="cba-rule"/>
</section><section class="cba-section cba-flow-card"><h2 class="cba-heading cba-h2" id="12-data-flow-tổng-quan"><span class="cba-section-pill">PIPELINE</span>12. Data Flow Tổng Quan</h2>
<pre class="cba-code"><code>┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 0: CONTRACT                                                           │
│  configs/base/coursework_contract.json → contracts/coursework.py             │
│  → artifacts/contracts/coursework_contract.{json,sha256}                      │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: ENVIRONMENT                                                        │
│  utils.environment.py → requirements_freeze.txt, environment_report.json    │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASES 2-4: RAW DATA                                                        │
│  data.acquisition → data.schema → data.temporal                              │
│  Input: link/raw_data/energydata_complete.csv                                │
│  Output: artifacts/{acquisition, schema, temporal}/                           │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASES 5-7: PROCESSING                                                      │
│  data.splitting → data.features → data.feature_sets                          │
│  → link/interim/.../energydata_feature_engineered_v1.csv                     │
│  → artifacts/{splits, features, feature_sets}/                                │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASES 8-11: SCALING + WINDOWS + DATALOADERS                                 │
│  data.scaling → data.windows → data.datasets                                 │
│  Output: artifacts/{scaling, scalers, windows, dataloaders}/                  │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASES 12-22: METRICS + MODELS + BASELINES                                   │
│  evaluation.metrics → experiments.registry → baselines.persistence           │
│  → models.{lstm_regressor, transformer_regressor}                            │
│  → attention.verification → sanity.forward_sanity → training.engine           │
│  → baselines.{lstm_baseline, transformer_b0} → diagnostics                    │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASES 23-41: HYPERPARAMETER SWEEPS (S1-S19)                                 │
│  sweeps.* → artifacts/sweeps/S*/                                              │
│  Each sweep: run variants → pick winner by val RMSE → update reference       │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 42: CANDIDATE SYNTHESIS                                                │
│  experiments.phase_execution.py → scripts/p42_candidate_synthesis.py        │
│  Output: artifacts/candidate_synthesis/                                       │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 43: LSTM TUNING                                                        │
│  lstm_tuning.* (5 stages: lt1-lt5)                                            │
│  → scripts/p43_lstm_tuning.py + 5 harness/sim scripts                        │
│  → artifacts/lstm_tuning/                                                     │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 44: ROLLING-ORIGIN ROBUSTNESS                                          │
│  rolling_origin.{folds, refit_engine, pipeline, scaling, ...}                  │
│  → scripts/p44_rolling_origin.py                                              │
│  → 5 folds × candidates → recommended Transformer                            │
│  → artifacts/rolling_origin/ (39 O44.* artifacts)                              │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 45: FINAL MODEL LOCK (NO-TRAIN)                                        │
│  final_model_lock.{candidate_lock, recipe, lineage, fingerprints}              │
│  → scripts/p45_final_model_lock.py                                            │
│  → artifacts/final_model_lock/ (38 O45.* artifacts)                            │
│  → FREEZE config: architecture, optimizer, loss, scaler, RevIN, seeds, ...   │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 46: THREE-SEED FINAL RUNS (TRAIN + VAL, FINAL_REFIT)                   │
│  data.final_dev + scaling.final_scaling                                        │
│  → scripts/p46_three_seed_runs.py + 5 corrective scripts                      │
│  → artifacts/three_seed_final_runs/official_checkpoints/seed_{42,123,2026}/*.pt │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  HISTORICAL V1 PHASE 47: FINAL TEST EVALUATION                               │
│  final_test_evaluation.{evaluation, checkpoint_loader, scaler_loader, ...}     │
│  → scripts/p47_final_test_evaluation.py                                       │
│  → Load Phase 46 checkpoints → Eval on Test (FINAL_TEST_POP-v1, L72)         │
│  → artifacts/final_test/ (RMSE, MAE, R² for 3 seeds + mean ± std)            │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  HISTORICAL V1 PHASES 48-59: ANALYSIS (READ-ONLY)                            │
│  analysis.* (12 subpackages)                                                  │
│  → Each phase: read Phase 47 predictions → produce figures/tables/findings   │
│  → reporting.* renders HTML dashboards                                       │
└──────────────────────────────────────────────────────────────────────────────┘
</code></pre>
<h3 class="cba-heading cba-h3" id="121-modelimprovementv2-extension-hiện-tại">12.1. MODEL_IMPROVEMENT_V2 extension hiện tại</h3>
<pre class="cba-code"><code class="language-text">E01–E20 development experiments
→ Step 14A equal-weight seed ensemble
→ Step 14B fixed persistence blend audit
→ Step 16 full pre-Test final refit and immutable lock
→ Step 17 POST_HOC_V2_BENCHMARK
→ reporting-only V2 analyses from frozen predictions
→ Phase 47–51 and 58–59 presentation
</code></pre>
<p>Final policy và lineage nằm tại <a class="cba-link" href="../artifacts/model_improvement_v2/final_model_lock/v2_final_model_lock.json"><code class="cba-inline-code">v2_final_model_lock.json</code></a>. Benchmark/MAPE nằm tại <a class="cba-link" href="../artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics_with_mape.json"><code class="cba-inline-code">step17_metrics_with_mape.json</code></a>. Reporting sources được khóa bởi <a class="cba-link" href="../artifacts/model_improvement_v2/final_reporting_analysis/manifest.json"><code class="cba-inline-code">final_reporting_analysis/manifest.json</code></a>.</p>
<hr class="cba-rule"/>
</section><section class="cba-section cba-design-card"><h2 class="cba-heading cba-h2" id="13-design-patterns--solid"><span class="cba-section-pill">DESIGN</span>13. Design Patterns &amp; SOLID</h2>
<h3 class="cba-heading cba-h3" id="131-design-patterns-identified">13.1. Design Patterns Identified</h3>
<p>| Pattern | Where Applied | How |
|---|---|---|
| <strong>Single Source of Truth</strong> | <a class="cba-link cba-path-link" href="../configs/base/coursework_contract.json"><code class="cba-inline-code">configs/base/coursework_contract.json</code></a> | One JSON drives all phases; SHA-256 fingerprint |
| <strong>Immutable Dataclass</strong> | Everywhere | <code class="cba-inline-code">@dataclass(frozen=True)</code> for configs (TransformerModelConfig, DatasetConfig, PredictionBundle, MetricResult, PhaseState, SweepPhaseSpec, FoldDefinition, ...) |
| <strong>Factory</strong> | Model/Dataset construction | <code class="cba-inline-code">build_reference_transformer_config()</code>, <code class="cba-inline-code">build_model_from_run_config()</code>, <code class="cba-inline-code">build_dataset_suite()</code>, <code class="cba-inline-code">build_dataloader()</code>, <code class="cba-inline-code">build_test_evaluation_loader()</code>, <code class="cba-inline-code">build_final_dev_dataset()</code> |
| <strong>Strategy</strong> | Loss/activation/pooling/scaling selection | Enum-driven: <code class="cba-inline-code">criterion_config()</code>, <code class="cba-inline-code">_resolve_activation()</code>, target_scaling YS0/YS1 |
| <strong>Registry</strong> | Experiment tracking | <code class="cba-inline-code">ExperimentRegistry</code> (JSONL + CSV), <a class="cba-link cba-path-link" href="../artifacts/experiments/experiment_families.csv"><code class="cba-inline-code">experiment_families.csv</code></a>, register_run/start_run/complete_run/fail_run |
| <strong>State Machine</strong> | Run lifecycle | <code class="cba-inline-code">RunStatus</code> (PLANNED→REGISTERED→RUNNING→{COMPLETED,FAILED,CANCELLED}→{INVALIDATED,ARCHIVED}), <code class="cba-inline-code">STATUS_TRANSITIONS</code> |
| <strong>Template Method</strong> | Every phase | <code class="cba-inline-code">materialize_phase_N()</code> family — verify upstream → build artifacts → write sign-off |
| <strong>Manifest + Sign-off</strong> | Universal artifact shape | <code class="cba-inline-code">{phase_N_signoff.json, _manifest.json, _contract.json, _discrepancies.json, _report.md}</code> |
| <strong>Single-write-once-or-verify</strong> | All artifact writes | <code class="cba-inline-code">utils.artifacts.write_bytes_once_or_verify()</code>, <code class="cba-inline-code">write_json_once_or_verify()</code> (strict), <code class="cba-inline-code">*_permissive</code> variants for metadata-only updates |
| <strong>Atomic Write + fsync</strong> | Crash-safe I/O | <code class="cba-inline-code">utils.artifacts.atomic_write_bytes()</code> — tempfile + fsync + os.replace |
| <strong>Test Firewall</strong> | Data leakage prevention | <code class="cba-inline-code">validate_evaluation_access()</code>, <code class="cba-inline-code cba-code-yellow">TargetAccessMode.TEST_LOCKED</code>, <code class="cba-inline-code">PHASE_47_AUTHORIZATION</code> |
| <strong>FINAL_REFIT Mode</strong> | Phase 46 train-on-TRAIN+VAL | <code class="cba-inline-code">TrainingEngine.train(final_refit_mode=True)</code>, <code class="cba-inline-code">EvaluationMode.FINAL_DEV_DIAGNOSTIC</code> |
| <strong>Fold-local Scalers</strong> | Phase 44 robustness | <code class="cba-inline-code">rolling_origin.scaling.{fit_fold_a_x_scaler, fit_fold_b_x_scaler, build_bundle}</code> |
| <strong>NO-TRAIN Orchestration</strong> | Phases 45, 47 | Governance phases with <code class="cba-inline-code">optimizer_steps=0, new_test_runs=0</code> |
| <strong>Population Fingerprinting</strong> | Cross-phase consistency | <code class="cba-inline-code">derive_population_fingerprint()</code>, <code class="cba-inline-code">MetricPopulationContext</code> |
| <strong>Heartbeat</strong> | Training monitoring | <code class="cba-inline-code">engine.py:SWEEP_HEARTBEAT_PATH</code>, external watchdog |
| <strong>Wall-clock Timeout</strong> | Sweep safety | <code class="cba-inline-code">install_training_timeout()</code> / <code class="cba-inline-code">disable_training_timeout()</code>, POSIX SIGALRM |
| <strong>Determinism</strong> | Reproducibility | <code class="cba-inline-code">set_seed()</code>, <code class="cba-inline-code cba-code-blue">configure_reproducibility("D0"|"D1")</code>, seed=42 default |</p>
<h3 class="cba-heading cba-h3" id="132-solid-principles">13.2. SOLID Principles</h3>
<div class="cba-table-wrap"><table class="cba-table">
<thead>
<tr>
<th>Principle</th>
<th>Adherence</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>S (Single Responsibility)</strong></td>
<td>✓ Strong: mỗi module 1 concern (metrics.py = metrics only, datasets.py = datasets only)</td>
</tr>
<tr>
<td><strong>O (Open/Closed)</strong></td>
<td>✓ Extends via enums (BoundaryProtocol, TargetAccessMode, EvaluationMode, FailureType) without modifying consumers</td>
</tr>
<tr>
<td><strong>L (Liskov Substitution)</strong></td>
<td>✓ LSTMRegressor và TransformerRegressor cùng <code class="cba-inline-code">forward(x)</code> signature; symmetric <code class="cba-inline-code">materialize_phase_15/16</code></td>
</tr>
<tr>
<td><strong>I (Interface Segregation)</strong></td>
<td>✓ SequenceWindowDataset returns minimal dicts (TEST: <code class="cba-inline-code">{x, sample_idx}</code>, TRAIN/VAL: <code class="cba-inline-code">{x, y_model, y_raw_wh}</code>)</td>
</tr>
<tr>
<td><strong>D (Dependency Inversion)</strong></td>
<td>✓ High-level (<code class="cba-inline-code">final_test_evaluation</code>, <code class="cba-inline-code">rolling_origin</code>) depend on abstractions (ExperimentRegistry, MetricResult) not concretes</td>
</tr>
</tbody>
</table></div>
<h3 class="cba-heading cba-h3" id="133-configuration-injection">13.3. Configuration Injection</h3>
<ul>
<li>Mọi hyperparameter đọc từ <a class="cba-link cba-path-link" href="../configs/base/coursework_contract.json"><code class="cba-inline-code">configs/base/coursework_contract.json</code></a> hoặc per-phase sign-off.</li>
<li>KHÔNG hard-code constants trong code.</li>
<li>Runtime configs (<code class="cba-inline-code">run_config.json</code> ở <code class="cba-inline-code">artifacts/runs/&lt;RUN_ID&gt;/</code>) validated against <code class="cba-inline-code">EXPERIMENT_FAMILIES</code>.</li>
</ul>
<hr class="cba-rule"/>
</section><section class="cba-section cba-repro-card"><h2 class="cba-heading cba-h2" id="14-reproducibility-story"><span class="cba-section-pill">REPRODUCIBILITY</span>14. Reproducibility Story</h2>
<h3 class="cba-heading cba-h3" id="141-determinism-layers">14.1. Determinism Layers</h3>
<ol>
<li><strong>Seed Layer:</strong> <code class="cba-inline-code">utils/reproducibility.set_seed(seed=42)</code> cho Python, NumPy, PyTorch (CPU + CUDA), DataLoader workers.</li>
<li><strong>Mode Layer:</strong> <code class="cba-inline-code cba-code-blue">configure_reproducibility(mode="D0")</code> — fully deterministic (cuDNN benchmark off, deterministic algorithms).</li>
<li><strong>Hardware Layer:</strong> <code class="cba-inline-code">select_device()</code> returns CPU/MPS/CUDA deterministically.</li>
<li><strong>Contract Layer:</strong> Single SHA-256 fingerprint per phase sign-off.</li>
<li><strong>Artifact Layer:</strong> Every artifact SHA-256 verified on write (single-write-once-or-verify).</li>
</ol>
<h3 class="cba-heading cba-h3" id="142-seed-strategy">14.2. Seed Strategy</h3>
<ul>
<li><strong>DEVELOPMENT_SEED:</strong> 42 (cho tất cả dev work, sweeps, baseline runs).</li>
<li><strong>FINAL_SEEDS:</strong> (42, 123, 2026) — chỉ dùng cho Phase 46 three-seed FINAL_REFIT.</li>
</ul>
<h3 class="cba-heading cba-h3" id="143-reproducibility-tests">14.3. Reproducibility Tests</h3>
<ul>
<li><code class="cba-inline-code">tests/integration/test_phase_46_cache_exclusion.py</code> (198 LOC) — verify cache exclusion.</li>
<li><code class="cba-inline-code">tests/unit/test_phase46_verification.py</code> (290 LOC) — verify Phase 46 reproducibility.</li>
<li><a class="cba-link cba-path-link" href="../tests/unit/test_reproducibility.py"><code class="cba-inline-code">tests/unit/test_reproducibility.py</code></a> — direct seed determinism tests.</li>
</ul>
<h3 class="cba-heading cba-h3" id="144-failure-recovery">14.4. Failure Recovery</h3>
<ul>
<li><code class="cba-inline-code">utils.environment.recover_environment_revision()</code> — move existing artifacts to <code class="cba-inline-code">_history/&lt;UTC_stamp&gt;/</code>.</li>
<li>Per-phase <code class="cba-inline-code">recover_X()</code> functions cho selective rollback.</li>
<li><code class="cba-inline-code">RERUN_REASONS</code> registry trong <code class="cba-inline-code">experiments/registry.py</code> cho audit trail.</li>
</ul>
<hr class="cba-rule"/>
</section><section class="cba-section cba-modules-card"><h2 class="cba-heading cba-h2" id="15-key-python-modules---detailed"><span class="cba-section-pill">MODULES</span>15. Key Python Modules - Detailed</h2>
<h3 class="cba-heading cba-h3" id="151-srccourseworkutilsartifactspy-215-loc">15.1. <a class="cba-link cba-path-link" href="../src/course_work/utils/artifacts.py"><code class="cba-inline-code">src/course_work/utils/artifacts.py</code></a> (215 LOC)</h3>
<p><strong>Purpose:</strong> Single I/O entry point. Tất cả writes phải qua đây.</p>
<p><strong>Key APIs:</strong></p>
<ul>
<li><code class="cba-inline-code">get_project_root()</code> — <code class="cba-inline-code">parents[3]</code> từ file location</li>
<li><code class="cba-inline-code">canonical_json_bytes()</code> — sorted JSON bytes</li>
<li><code class="cba-inline-code">sha256_bytes()</code>, <code class="cba-inline-code">sha256_file()</code></li>
<li><code class="cba-inline-code">atomic_write_bytes()</code> — tempfile + fsync + os.replace</li>
<li><code class="cba-inline-code">write_bytes_once_or_verify()</code> — strict: re-write fail unless identical</li>
<li><code class="cba-inline-code">write_bytes_once_or_verify_permissive()</code> — allow metadata updates with <code class="cba-inline-code">compare_keys</code></li>
<li><code class="cba-inline-code">_strip_provenance_metadata()</code> — strip <code class="cba-inline-code">created_at</code> for stable fingerprints</li>
<li><code class="cba-inline-code">write_json_once_or_verify()</code>, <code class="cba-inline-code">write_text_once_or_verify()</code> (CSV-aware)</li>
</ul>
<h3 class="cba-heading cba-h3" id="152-srccourseworkutilsenvironmentpy-423-loc">15.2. <a class="cba-link cba-path-link" href="../src/course_work/utils/environment.py"><code class="cba-inline-code">src/course_work/utils/environment.py</code></a> (423 LOC)</h3>
<p><strong>Purpose:</strong> Environment capture, device selection, dependency freeze.</p>
<p><strong>Key APIs:</strong></p>
<ul>
<li><code class="cba-inline-code">CORE_DISTRIBUTIONS</code>, <code class="cba-inline-code">STABLE_ENVIRONMENT_FIELDS</code></li>
<li><code class="cba-inline-code">select_device()</code> — CUDA &gt; MPS &gt; CPU</li>
<li><code class="cba-inline-code">resolve_kernel_contract()</code></li>
<li><code class="cba-inline-code">environment_inventory()</code>, <code class="cba-inline-code">environment_identity()</code>, <code class="cba-inline-code">environment_identity_differences()</code></li>
<li><code class="cba-inline-code">device_smoke_test()</code></li>
<li><code class="cba-inline-code">dependency_freeze()</code> — writes <code class="cba-inline-code">requirements_freeze.txt</code></li>
<li><code class="cba-inline-code">materialize_phase_1()</code> — Phase 1 entry</li>
<li><code class="cba-inline-code">recover_environment_revision()</code> — <code class="cba-inline-code">_history/</code> snapshot</li>
</ul>
<h3 class="cba-heading cba-h3" id="153-srccourseworkutilsreproducibilitypy-76-loc">15.3. <a class="cba-link cba-path-link" href="../src/course_work/utils/reproducibility.py"><code class="cba-inline-code">src/course_work/utils/reproducibility.py</code></a> (76 LOC)</h3>
<p><strong>Purpose:</strong> Seed &amp; determinism.</p>
<p><strong>Constants:</strong></p>
<ul>
<li><code class="cba-inline-code">DEVELOPMENT_SEED = 42</code></li>
<li><code class="cba-inline-code">FINAL_SEEDS = (42, 123, 2026)</code></li>
</ul>
<p><strong>Key APIs:</strong></p>
<ul>
<li><code class="cba-inline-code">set_seed(seed)</code> — Python + NumPy + PyTorch (CPU + CUDA) + DataLoader workers</li>
<li><code class="cba-inline-code cba-code-blue">configure_reproducibility(mode="D0"|"D1")</code></li>
<li><code class="cba-inline-code">seed_worker(worker_id)</code> — DataLoader worker init</li>
<li><code class="cba-inline-code">build_torch_generator(seed)</code></li>
<li><code class="cba-inline-code">randomness_smoke_test()</code></li>
</ul>
<h3 class="cba-heading cba-h3" id="154-srccourseworkcontractscourseworkpy-183-loc">15.4. <a class="cba-link cba-path-link" href="../src/course_work/contracts/coursework.py"><code class="cba-inline-code">src/course_work/contracts/coursework.py</code></a> (183 LOC)</h3>
<p><strong>Purpose:</strong> Phase 0 contract loader/validator.</p>
<p><strong>Key APIs:</strong></p>
<ul>
<li><code class="cba-inline-code">EXPECTED_OPTION_IDS</code> — set per group</li>
<li><code class="cba-inline-code">load_coursework_contract(path=None)</code></li>
<li><code class="cba-inline-code">validate_coursework_contract(contract)</code> → <code class="cba-inline-code">tuple[str, ...]</code></li>
<li><code class="cba-inline-code">coursework_contract_fingerprint(contract)</code> — SHA-256</li>
<li><code class="cba-inline-code">materialize_phase_0(project_root=None)</code></li>
</ul>
<h3 class="cba-heading cba-h3" id="155-srccourseworkdatadatasetspy-1273-loc">15.5. <a class="cba-link cba-path-link" href="../src/course_work/data/datasets.py"><code class="cba-inline-code">src/course_work/data/datasets.py</code></a> (1,273 LOC)</h3>
<p><strong>Purpose:</strong> Phase 11 DataLoader construction + Test firewall.</p>
<p><strong>Constants:</strong></p>
<ul>
<li><code class="cba-inline-code">DATALOADER_VERSION = "DATALOADERS-v1"</code></li>
<li><code class="cba-inline-code">TargetAccessMode</code> enum: TRAIN, VALIDATION, TEST_LOCKED, TEST_EVALUATION, FINAL_DEV</li>
</ul>
<p><strong>Key Classes:</strong></p>
<ul>
<li><code class="cba-inline-code">DatasetConfig</code> (frozen)</li>
<li><code class="cba-inline-code">LoaderConfig</code> (frozen)</li>
<li><code class="cba-inline-code">SequenceWindowDataset</code> (map-style PyTorch Dataset)</li>
</ul>
<p><strong>Key Builders:</strong></p>
<ul>
<li><code class="cba-inline-code">build_dataset_suite()</code> — TRAIN/VAL/TEST_LOCKED</li>
<li><code class="cba-inline-code">build_test_evaluation_dataset()</code> — for Phase 47</li>
<li><code class="cba-inline-code">build_final_dev_dataset()</code> — for Phase 46</li>
<li><code class="cba-inline-code">build_dataloader()</code>, <code class="cba-inline-code">build_train_validation_loaders()</code>, <code class="cba-inline-code">build_test_locked_loader()</code>, <code class="cba-inline-code">build_test_evaluation_loader()</code></li>
</ul>
<p><strong>Test Firewall:</strong> Mỗi builder enforce access mode. TEST_LOCKED chỉ accessible khi <code class="cba-inline-code">PHASE_47_AUTHORIZATION</code> granted.</p>
<h3 class="cba-heading cba-h3" id="156-srccourseworkmodelstransformerregressorpy-507-loc">15.6. <a class="cba-link cba-path-link" href="../src/course_work/models/transformer_regressor.py"><code class="cba-inline-code">src/course_work/models/transformer_regressor.py</code></a> (507 LOC)</h3>
<p><strong>Purpose:</strong> Phase 16 Transformer architecture.</p>
<p><strong>Constants:</strong></p>
<ul>
<li><code class="cba-inline-code">TRANSFORMER_IMPL_VERSION = "TRANSFORMER_IMPL-v1"</code></li>
<li><code class="cba-inline-code">PHASE_VERSION = "PHASE-16-v1"</code></li>
</ul>
<p><strong>Key Classes:</strong></p>
<ul>
<li><code class="cba-inline-code">TransformerModelConfig</code> (frozen): d_model=64, num_heads=4, num_layers=2, ffn_dim=128, dropout=0.1, GELU, LAST_STEP, SINUSOIDAL, attention_aware=True, post-norm</li>
<li><code class="cba-inline-code">validate_transformer_config()</code> — strict shape/dtype validation</li>
<li><code class="cba-inline-code">TransformerRegressor(nn.Module)</code> — <code class="cba-inline-code">forward()</code> and <code class="cba-inline-code">forward_with_attention()</code> (returns per-head [B,H,L,L])</li>
<li><code class="cba-inline-code">build_reference_transformer_config()</code>, <code class="cba-inline-code">materialize_phase_16()</code></li>
</ul>
<h3 class="cba-heading cba-h3" id="157-srccourseworkmodelslstmregressorpy-466-loc">15.7. <a class="cba-link cba-path-link" href="../src/course_work/models/lstm_regressor.py"><code class="cba-inline-code">src/course_work/models/lstm_regressor.py</code></a> (466 LOC)</h3>
<p><strong>Purpose:</strong> Phase 15 LSTM architecture.</p>
<p><strong>Constants:</strong></p>
<ul>
<li><code class="cba-inline-code">LSTM_IMPL_VERSION = "LSTM_IMPL-v1"</code></li>
</ul>
<p><strong>Config:</strong></p>
<ul>
<li>Unidirectional only (<code class="cba-inline-code">bidirectional=False</code>)</li>
<li><code class="cba-inline-code">batch_first=True</code></li>
<li><code class="cba-inline-code">LAST_STEP</code> pooling</li>
<li>Stateless zero-init hidden state</li>
</ul>
<h3 class="cba-heading cba-h3" id="158-srccourseworkmodelsrevinpy-273-loc">15.8. <a class="cba-link cba-path-link" href="../src/course_work/models/revin.py"><code class="cba-inline-code">src/course_work/models/revin.py</code></a> (273 LOC)</h3>
<p><strong>Purpose:</strong> RevIN (Reversible Instance Normalization) for Phase 40.</p>
<p><strong>Constants:</strong></p>
<ul>
<li><code class="cba-inline-code">REVIN_EPS = 1e-5</code></li>
<li>Affine, target-selective (RN1 contract)</li>
</ul>
<p><strong>Key Classes:</strong></p>
<ul>
<li><code class="cba-inline-code">TargetSelectiveRevIN(nn.Module)</code></li>
<li><code class="cba-inline-code">RevINScope</code> dataclass</li>
</ul>
<h3 class="cba-heading cba-h3" id="159-srccourseworktrainingenginepy-615-loc">15.9. <a class="cba-link cba-path-link" href="../src/course_work/training/engine.py"><code class="cba-inline-code">src/course_work/training/engine.py</code></a> (615 LOC)</h3>
<p><strong>Purpose:</strong> Phase 19 generic training loop.</p>
<p><strong>Key Classes:</strong></p>
<ul>
<li><code class="cba-inline-code">TrainingResult</code> dataclass</li>
<li><code class="cba-inline-code">EarlyStopping</code> (MIN mode only, patience-based)</li>
<li><code class="cba-inline-code">TrainingEngine</code> — main loop with heartbeat, non-finite-gradient guard, gradient clip, optional FINAL_REFIT mode</li>
</ul>
<p><strong>Key Functions:</strong></p>
<ul>
<li><code class="cba-inline-code">build_model_from_run_config()</code> — factory</li>
<li><code class="cba-inline-code">persist_run_artifacts()</code> — save checkpoint (best+last), history, predictions, metrics</li>
<li><code class="cba-inline-code">install_training_timeout()</code> / <code class="cba-inline-code">disable_training_timeout()</code> — POSIX SIGALRM</li>
<li><code class="cba-inline-code">metric_unit_for()</code> — helper</li>
</ul>
<h3 class="cba-heading cba-h3" id="1510-srccourseworkevaluationmetricspy-1182-loc">15.10. <a class="cba-link cba-path-link" href="../src/course_work/evaluation/metrics.py"><code class="cba-inline-code">src/course_work/evaluation/metrics.py</code></a> (1,182 LOC)</h3>
<p><strong>Purpose:</strong> Phase 12 shared metrics.</p>
<p><strong>Enums:</strong></p>
<ul>
<li><code class="cba-inline-code">EvaluationMode</code>: TRAIN_DIAGNOSTIC, VALIDATION, FINAL_TEST, FINAL_DEV_DIAGNOSTIC</li>
</ul>
<p><strong>Key Classes:</strong></p>
<ul>
<li><code class="cba-inline-code">EvaluationContext</code></li>
<li><code class="cba-inline-code">MetricPopulationContext</code> (Phase 44 fold-aware)</li>
<li><code class="cba-inline-code">PredictionBundle</code></li>
<li><code class="cba-inline-code">MetricResult</code></li>
<li><code class="cba-inline-code">SupplementaryMapeResult</code></li>
</ul>
<p><strong>Key Functions:</strong></p>
<ul>
<li><code class="cba-inline-code">validate_evaluation_access()</code> — strict split/mode firewall</li>
<li><code class="cba-inline-code">derive_population_fingerprint()</code> — canonical</li>
<li><code class="cba-inline-code">convert_predictions_to_wh()</code> — inverse scale</li>
<li><code class="cba-inline-code">compute_mae_wh()</code>, <code class="cba-inline-code">compute_rmse_wh()</code>, <code class="cba-inline-code">compute_r2()</code></li>
<li><code class="cba-inline-code">compute_regression_metrics()</code> — central function</li>
<li><code class="cba-inline-code">build_mape_metric_contract()</code>, <code class="cba-inline-code">compute_mape_pct()</code></li>
<li><code class="cba-inline-code">compare_to_baseline()</code>, <code class="cba-inline-code">materialize_phase_12()</code></li>
</ul>
<h3 class="cba-heading cba-h3" id="1511-srccourseworkbaselinespersistencepy-1161-loc">15.11. <a class="cba-link cba-path-link" href="../src/course_work/baselines/persistence.py"><code class="cba-inline-code">src/course_work/baselines/persistence.py</code></a> (1,161 LOC)</h3>
<p><strong>Purpose:</strong> Phase 14 persistence baseline.</p>
<p><strong>Constants:</strong></p>
<ul>
<li><code class="cba-inline-code">PERSISTENCE_VERSION = "PERSISTENCE-v1"</code></li>
<li><code class="cba-inline-code">TEST_ACCESS_POLICY = "LOCKED_UNTIL_PHASE_47"</code></li>
</ul>
<p><strong>Key Classes:</strong></p>
<ul>
<li><code class="cba-inline-code">PersistenceConfig</code></li>
<li><code class="cba-inline-code">PersistencePreparedData</code></li>
<li><code class="cba-inline-code">PersistenceEvaluationResult</code></li>
</ul>
<p><strong>Key Functions:</strong></p>
<ul>
<li><code class="cba-inline-code">predict_persistence()</code> — pure: $\hat{y}[t+1] = y[t]$ với 4 test cases</li>
<li><code class="cba-inline-code">prepare_validation_persistence_data()</code></li>
<li><code class="cba-inline-code">run_persistence_contract_tests()</code> — 8 unit tests + lookback invariance (L36/L72/L144)</li>
<li><code class="cba-inline-code">materialize_phase_14()</code></li>
</ul>
<h3 class="cba-heading cba-h3" id="1512-srccourseworkexperimentsregistrypy-2191-loc">15.12. <a class="cba-link cba-path-link" href="../src/course_work/experiments/registry.py"><code class="cba-inline-code">src/course_work/experiments/registry.py</code></a> (2,191 LOC)</h3>
<p><strong>Purpose:</strong> Phase 13 experiment tracking.</p>
<p><strong>Constants:</strong></p>
<ul>
<li><code class="cba-inline-code">EXPERIMENT_VERSION = "EXPERIMENTS-v1"</code></li>
<li><code class="cba-inline-code">RECORD_SCHEMA_VERSION = 1</code></li>
<li><code class="cba-inline-code cba-code-yellow">FAILURE_TAXONOMY_VERSION = "FAILURES-v1"</code></li>
<li><code class="cba-inline-code">FEATURE_VARIANTS = {"FS0_TF0", "FS0_TF1", "FS1_TF0", "FS1_TF1", "FS2_TF0", "FS2_TF1"}</code></li>
<li><code class="cba-inline-code">MODEL_FAMILIES = {"PERSISTENCE", "LSTM", "TRANSFORMER_ENCODER"}</code></li>
<li><code class="cba-inline-code">EXPERIMENT_FAMILIES</code> — 26 entries</li>
<li><code class="cba-inline-code">RERUN_REASONS</code> — incl. <code class="cba-inline-code">PHASE44_CORRECTIVE_RERUN</code>, <code class="cba-inline-code">PHASE46_CORRECTIVE_RERUN</code>, <code class="cba-inline-code">PHASE46_HARD_STOP_PROBE</code></li>
</ul>
<p><strong>Enums:</strong></p>
<ul>
<li><code class="cba-inline-code">RunStatus</code> (PLANNED→...→COMPLETED/FAILED/CANCELLED→INVALIDATED/ARCHIVED)</li>
<li><code class="cba-inline-code">ExecutionType</code></li>
<li><code class="cba-inline-code">FailureType</code> (16 codes)</li>
<li><code class="cba-inline-code">ArtifactType</code> (13 codes)</li>
<li><code class="cba-inline-code">STATUS_TRANSITIONS</code></li>
</ul>
<p><strong>Key Class:</strong> <code class="cba-inline-code">ExperimentRegistry</code></p>
<ul>
<li>Manages: <code class="cba-inline-code">experiment_registry.jsonl</code>, <code class="cba-inline-code">experiment_registry.csv</code>, <a class="cba-link cba-path-link" href="../artifacts/experiments/experiment_families.csv"><code class="cba-inline-code">experiment_families.csv</code></a>, <code class="cba-inline-code">run_artifact_registry.csv</code></li>
<li>APIs: <code class="cba-inline-code">register_run()</code>, <code class="cba-inline-code">start_run()</code>, <code class="cba-inline-code">complete_run()</code>, <code class="cba-inline-code">fail_run()</code>, <code class="cba-inline-code">cancel_run()</code>, <code class="cba-inline-code">validate_registry()</code></li>
</ul>
<h3 class="cba-heading cba-h3" id="1513-srccourseworkexperimentsphaseexecutionpy-671-loc">15.13. <a class="cba-link cba-path-link" href="../src/course_work/experiments/phase_execution.py"><code class="cba-inline-code">src/course_work/experiments/phase_execution.py</code></a> (671 LOC)</h3>
<p><strong>Purpose:</strong> Sweep phase execution orchestration.</p>
<p><strong>Enums:</strong></p>
<ul>
<li><code class="cba-inline-code">PhaseState</code> (9 states: VALID_REUSABLE, LOG_MISSING, LOG_STALE, DERIVED_ARTIFACT_MISSING, CONDITION_INCOMPLETE, SIGNOFF_INVALID, UPSTREAM_INVALID, ENVIRONMENT_INVALID, RUNNING, FAILED)</li>
<li><code class="cba-inline-code">PhaseAction</code> (6 actions: PROCEED, SKIP, RE_RUN, RECOVER, ABORT, FORCE)</li>
</ul>
<p><strong>Key Class:</strong></p>
<ul>
<li><code class="cba-inline-code">SweepPhaseSpec</code> — declares conditions, prerequisite paths, winner/reference files, log file</li>
</ul>
<p><strong>Dict:</strong> <code class="cba-inline-code">SWEEP_PHASE_SPECS</code> — specs cho phases 23-41.</p>
<h3 class="cba-heading cba-h3" id="1514-srccourseworkfinaltestevaluationinitpy-hard-locked-constants">15.14. <a class="cba-link cba-path-link" href="../src/course_work/final_test_evaluation/__init__.py"><code class="cba-inline-code">src/course_work/final_test_evaluation/__init__.py</code></a> (hard-locked constants)</h3>
<p><strong>Purpose:</strong> Phase 47 final test gate constants.</p>
<p><strong>Hard-locked Constants:</strong></p>
<ul>
<li><code class="cba-inline-code">LOCKED_CANDIDATE = "TR_C2_ALT_LOOKBACK"</code></li>
<li><code class="cba-inline-code">LOCKED_LOOKBACK = 72</code></li>
<li><code class="cba-inline-code">LOCKED_FEATURES = 33</code></li>
<li><code class="cba-inline-code">LOCKED_BOUNDARY_PROTOCOL = "WB0_CONTEXT_CARRY_OVER"</code></li>
<li><code class="cba-inline-code">LOCKED_SEEDS = [42, 123, 2026]</code></li>
<li><code class="cba-inline-code">LOCKED_CONFIG_FP = "585c5e79e6a1c8c4..."</code> (SHA-256)</li>
<li><code class="cba-inline-code">OFFICIAL_RUNS</code> — dict với checkpoint_sha256 per seed</li>
<li><code class="cba-inline-code">FINAL_SCALING</code> — dict với x/y_scaler_sha256</li>
<li><code class="cba-inline-code">FINAL_DEV_POP_FP</code>, <code class="cba-inline-code">FINAL_DEV_WINDOW_COUNT = 16630</code></li>
<li><code class="cba-inline-code">LSTM_TUNED_DEV</code> — metadata (known L36 vs L72 fairness caveat)</li>
<li><code class="cba-inline-code">PHASE_47_SPLIT = "TEST"</code></li>
<li><code class="cba-inline-code">LSTM_ELIGIBILITY_STATUSES</code></li>
</ul>
<hr class="cba-rule"/>
</section><section class="cba-section cba-anomaly-card"><h2 class="cba-heading cba-h2" id="16-anomalies--observations"><span class="cba-section-pill">ANOMALIES</span>16. Anomalies &amp; Observations</h2>
<h3 class="cba-heading cba-h3" id="161-anomalies">16.1. Anomalies</h3>
<div class="cba-table-wrap"><table class="cba-table">
<thead>
<tr>
<th>ID</th>
<th>Anomaly</th>
<th>Impact</th>
<th>Resolution</th>
</tr>
</thead>
<tbody>
<tr class="cba-anomaly-row">
<td><strong>A1</strong></td>
<td>Missing <code class="cba-inline-code">COURSE_WORK/README.md</code></td>
<td>Developer onboarding khó</td>
<td>Tạo file (TODO)</td>
</tr>
<tr class="cba-anomaly-row">
<td><strong>A2</strong></td>
<td><code class="cba-inline-code">data/</code> directory missing, only <a class="cba-link cba-path-link" href="../link"><code class="cba-inline-code">link/</code></a> exists</td>
<td>Code may fail if <code class="cba-inline-code">data/</code> not symlinked</td>
<td>Verify symlink or remap paths</td>
</tr>
<tr class="cba-anomaly-row">
<td><strong>A3</strong></td>
<td>Empty <a class="cba-link cba-path-link" href="../artifacts/scalers"><code class="cba-inline-code">artifacts/scalers/</code></a></td>
<td>Phase 9 scalers missing on disk</td>
<td>Regenerate from <code class="cba-inline-code">scaling/scaler_registry.json</code></td>
</tr>
<tr class="cba-anomaly-row">
<td><strong>A4</strong></td>
<td>Notebook phase numbering offset (+1) vs artifacts</td>
<td>Confusing docs</td>
<td>Standardize on artifact numbering (current_flow docs already follow)</td>
</tr>
<tr class="cba-anomaly-row">
<td><strong>A5</strong></td>
<td>Empty stubs: <code class="cba-inline-code">attention/{extraction,head_comparison,heatmaps,last_query}.py</code></td>
<td>0 bytes; real code in <code class="cba-inline-code">analysis/</code></td>
<td>Delete stubs or move logic</td>
</tr>
<tr class="cba-anomaly-row">
<td><strong>A6</strong></td>
<td>Two parallel <a class="cba-link cba-path-link" href="../scripts"><code class="cba-inline-code">scripts/</code></a> directories</td>
<td>Confusion (in-src vs root-level)</td>
<td>Document distinction</td>
</tr>
<tr class="cba-anomaly-row">
<td><strong>A7</strong></td>
<td>~50 disposable probe scripts (<code class="cba-inline-code">_phase4{4,5,6,7}_*.py</code>)</td>
<td>Pollute root <a class="cba-link cba-path-link" href="../scripts"><code class="cba-inline-code">scripts/</code></a></td>
<td>Move to <code class="cba-inline-code">_disposable/</code> or delete</td>
</tr>
<tr class="cba-anomaly-row">
<td><strong>A8</strong></td>
<td>Multiple <code class="cba-inline-code">_history/</code> directories</td>
<td>Spread across artifacts/</td>
<td>OK — intentional snapshot pattern</td>
</tr>
<tr class="cba-anomaly-row">
<td><strong>A10</strong></td>
<td>LSTM known fairness caveat</td>
<td>LSTM not eval-able on same pop as Transformer</td>
<td>Documented in <code class="cba-inline-code">final_test_evaluation/__init__.py:LSTM_TUNED_DEV</code></td>
</tr>
<tr class="cba-anomaly-row">
<td><strong>A11</strong></td>
<td>Manual rerun authorization registry</td>
<td>Same fingerprints registered multiple times</td>
<td>Intentional recovery pattern</td>
</tr>
<tr class="cba-anomaly-row">
<td><strong>A12</strong></td>
<td>FINAL_DEV vs FINAL_TEST_POP-v1 naming</td>
<td>Different populations</td>
<td>Documented, intentional</td>
</tr>
<tr class="cba-anomaly-row">
<td><strong>A13</strong></td>
<td><code class="cba-inline-code">__pycache__</code> operation-not-permitted</td>
<td>macOS sandbox artifact</td>
<td>Ignore</td>
</tr>
</tbody>
</table></div>
<h3 class="cba-heading cba-h3" id="162-large-modules-god-modules">16.2. Large Modules (god-modules)</h3>
<ul>
<li><code class="cba-inline-code">experiments/registry.py</code> — 2,191 LOC</li>
<li><code class="cba-inline-code">baselines/persistence.py</code> — 1,161 LOC</li>
<li><code class="cba-inline-code">data/datasets.py</code> — 1,273 LOC</li>
<li><code class="cba-inline-code">evaluation/metrics.py</code> — 1,182 LOC</li>
<li><code class="cba-inline-code">scripts/p43_lstm_tuning.py</code> — 2,882 LOC</li>
<li><code class="cba-inline-code">scripts/p40_prepare_revin.py</code> — 1,509 LOC</li>
<li><code class="cba-inline-code">scripts/p40_finalize.py</code> — 1,439 LOC</li>
<li><code class="cba-inline-code">scripts/p46_three_seed_runs.py</code> — 2,242 LOC</li>
<li><code class="cba-inline-code">scripts/p47_final_test_evaluation.py</code> — 891 LOC</li>
<li><code class="cba-inline-code">rolling_origin/pipeline.py</code> — 1,539 LOC</li>
<li><code class="cba-inline-code">rolling_origin/real_run.py</code> — 1,673 LOC</li>
<li><code class="cba-inline-code">final_test_evaluation/writers.py</code> — 1,815 LOC</li>
</ul>
<p><strong>Recommendation:</strong> Tương lai nên split thành các submodules nhỏ hơn. Hiện tại chấp nhận được do mỗi file có 1 concern rõ ràng.</p>
<h3 class="cba-heading cba-h3" id="163-emptydead-code">16.3. Empty/Dead Code</h3>
<ul>
<li>4 empty stubs in <code class="cba-inline-code">attention/</code></li>
<li>~50 disposable probe scripts in root <a class="cba-link cba-path-link" href="../scripts"><code class="cba-inline-code">scripts/</code></a></li>
<li>Empty <a class="cba-link cba-path-link" href="../artifacts/scalers"><code class="cba-inline-code">artifacts/scalers/</code></a> directory</li>
</ul>
<hr class="cba-rule"/>
</section><section class="cba-section cba-rules-card"><h2 class="cba-heading cba-h2" id="17-quy-ước--rule-base"><span class="cba-section-pill">RULES</span>17. Quy Ước &amp; Rule Base</h2>
<h3 class="cba-heading cba-h3" id="171-code-style--docsrulebaserulecodemd-">17.1. Code Style (<a class="cba-link cba-path-link" href="../docs/rule_base/rule_code.md"><code class="cba-inline-code">docs/rule_base/rule_code.md</code></a>)</h3>
<ul>
<li>Type hints required cho public APIs.</li>
<li>Frozen dataclasses cho configs.</li>
<li>Docstrings cho mọi module/class.</li>
<li>Snake_case cho functions/variables.</li>
<li>PascalCase cho classes.</li>
<li>ALL_CAPS cho constants.</li>
<li>Imports grouped: stdlib, third-party, local.</li>
</ul>
<h3 class="cba-heading cba-h3" id="172-architecture-rules--docsrulebasearchitecturerulemd-">17.2. Architecture Rules (<a class="cba-link cba-path-link" href="../docs/rule_base/architecture_rule.md"><code class="cba-inline-code">docs/rule_base/architecture_rule.md</code></a>)</h3>
<ul>
<li>Clean Architecture tầng (Models → Data → Evaluation → Experiments → Phase-specific).</li>
<li><code class="cba-inline-code">models/</code> KHÔNG import <code class="cba-inline-code">data/</code>, <code class="cba-inline-code">training/</code>, <code class="cba-inline-code">experiments/</code>.</li>
<li><code class="cba-inline-code">data/</code> KHÔNG import <code class="cba-inline-code">models/</code>, <code class="cba-inline-code">training/</code>.</li>
<li>Mọi I/O qua <code class="cba-inline-code">utils.artifacts</code>.</li>
<li>KHÔNG hard-code hyperparameters.</li>
<li>Test firewall: V1 chỉ mở Test tại historical Phase 47; V2 chỉ mở old Test tại Step 17 sau immutable Step 16 lock.</li>
</ul>
<h3 class="cba-heading cba-h3" id="173-phase-sign-off-pattern">17.3. Phase Sign-off Pattern</h3>
<pre class="cba-code"><code class="language-json">{
  "phase": 47,
  "status": "PASS",  // or "FAIL"
  "input_checksums": {"...": "sha256"},
  "output_checksums": {"...": "sha256"},
  "config_fingerprint": "sha256",
  "tests": [...],
  "warnings": [...],
  "discrepancies": [...],
  "created_at": "ISO timestamp"
}
</code></pre>
<h3 class="cba-heading cba-h3" id="174-artifact-naming">17.4. Artifact Naming</h3>
<ul>
<li><code class="cba-inline-code">phase_N_signoff.json</code> — gate</li>
<li><code class="cba-inline-code">phase_N_manifest.json</code> — checksums</li>
<li><code class="cba-inline-code">phase_N_contract.json</code> — config lock</li>
<li><code class="cba-inline-code">phase_N_report.md</code> — human-readable</li>
<li><code class="cba-inline-code">phase_N_discrepancies.json</code> — issues</li>
<li><code class="cba-inline-code">phase_N_summary.json</code> — quick metrics</li>
</ul>
<hr class="cba-rule"/>
</section><section class="cba-section cba-appendix-card"><h2 class="cba-heading cba-h2" id="18-phụ-lục"><span class="cba-section-pill">APPENDIX</span>18. Phụ Lục</h2>
<h3 class="cba-heading cba-h3" id="181-glossary">18.1. Glossary</h3>
<div class="cba-table-wrap"><table class="cba-table">
<thead>
<tr>
<th>Term</th>
<th>Definition</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Phase</strong></td>
<td>Một bước trong pipeline (0-59)</td>
</tr>
<tr>
<td><strong>Sign-off</strong></td>
<td>JSON gate file xác nhận phase hoàn thành</td>
</tr>
<tr>
<td><strong>Manifest</strong></td>
<td>JSON liệt kê inputs/outputs với checksums</td>
</tr>
<tr>
<td><strong>FINAL_REFIT</strong></td>
<td>Train mode train-on-TRAIN+VAL (no early stop)</td>
</tr>
<tr>
<td><strong>Test Firewall</strong></td>
<td>Cơ chế chặn truy cập Test data trước Phase 47</td>
</tr>
<tr>
<td><strong>Population Fingerprint</strong></td>
<td>SHA-256 định danh một population (fold/split)</td>
</tr>
<tr>
<td><strong>NO-TRAIN Phase</strong></td>
<td>Phase chỉ lock config/governance, không train</td>
</tr>
<tr>
<td><strong>Sweep</strong></td>
<td>Hyperparameter sweep qua nhiều conditions</td>
</tr>
<tr>
<td><strong>Winner</strong></td>
<td>Condition tốt nhất trong sweep (theo validation RMSE)</td>
</tr>
<tr>
<td><strong>Reference Update</strong></td>
<td>Update config cho sweep kế tiếp dựa trên winner</td>
</tr>
</tbody>
</table></div>
<h3 class="cba-heading cba-h3" id="182-quick-links">18.2. Quick Links</h3>
<div class="cba-table-wrap"><table class="cba-table">
<thead>
<tr>
<th>Document</th>
<th>Path</th>
</tr>
</thead>
<tbody>
<tr>
<td>Current Flow Summary</td>
<td><a class="cba-link cba-path-link" href="../docs/current_flow/CURRENT_FLOW_SUMMARY.md"><code class="cba-inline-code">docs/current_flow/CURRENT_FLOW_SUMMARY.md</code></a></td>
</tr>
<tr>
<td>Phase 1-33 Detail</td>
<td><a class="cba-link cba-path-link" href="../docs/current_flow/PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md"><code class="cba-inline-code">docs/current_flow/PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md</code></a></td>
</tr>
<tr>
<td>Phase 34-59 Detail</td>
<td><a class="cba-link cba-path-link" href="../docs/current_flow/PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md"><code class="cba-inline-code">docs/current_flow/PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md</code></a></td>
</tr>
<tr>
<td>Notebook Walkthrough</td>
<td><a class="cba-link cba-path-link" href="../docs/link&discussion_to_result/NOTEBOOK_CELLS_WALKTHROUGH.md"><code class="cba-inline-code">docs/link&amp;discussion_to_result/NOTEBOOK_CELLS_WALKTHROUGH.md</code></a></td>
</tr>
<tr>
<td>Code Style Rules</td>
<td><a class="cba-link cba-path-link" href="../docs/rule_base/rule_code.md"><code class="cba-inline-code">docs/rule_base/rule_code.md</code></a></td>
</tr>
<tr>
<td>Architecture Rules</td>
<td><a class="cba-link cba-path-link" href="../docs/rule_base/architecture_rule.md"><code class="cba-inline-code">docs/rule_base/architecture_rule.md</code></a></td>
</tr>
<tr>
<td>Main Contract</td>
<td><a class="cba-link cba-path-link" href="../configs/base/coursework_contract.json"><code class="cba-inline-code">configs/base/coursework_contract.json</code></a></td>
</tr>
<tr>
<td>Notebook</td>
<td><a class="cba-link cba-path-link" href="../notebook_course_work/CourseWork.ipynb"><code class="cba-inline-code">notebook_course_work/CourseWork.ipynb</code></a></td>
</tr>
</tbody>
</table></div>
<h3 class="cba-heading cba-h3" id="183-current-verified-entry-points">18.3. Current verified entry points</h3>
<ul>
<li><a class="cba-link cba-notebook-link" href="../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Markdown links do not reliably deep-link to an exact notebook cell.">Open Notebook ↗</a></li>
<li><a class="cba-link" href="link&amp;discussion_to_result/NOTEBOOK_CELLS_WALKTHROUGH.md">NOTEBOOK_CELLS_WALKTHROUGH.md</a></li>
<li><a class="cba-link" href="current_flow/CURRENT_FLOW_SUMMARY.md">CURRENT_FLOW_SUMMARY.md</a></li>
<li><a class="cba-link" href="current_flow/PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md">PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md</a></li>
<li><a class="cba-link" href="current_flow/PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md">PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md</a></li>
<li><a class="cba-link" href="../artifacts/model_improvement_v2/final_model_lock/v2_final_model_lock.json">V2 final lock</a></li>
<li><a class="cba-link" href="../artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics_with_mape.json">Step 17 metrics with MAPE</a></li>
<li><a class="cba-link" href="../artifacts/model_improvement_v2/model_improvement_v2_final_closure.json">V2 final closure</a></li>
</ul>
<h3 class="cba-heading cba-h3" id="184-recommended-reading-order">18.4. Recommended Reading Order</h3>
<ol>
<li><strong>Bắt đầu:</strong> Đọc file này (<a class="cba-link cba-path-link" href="../docs/code_base_audit.md"><code class="cba-inline-code">code_base_audit.md</code></a>) → hiểu tổng quan.</li>
<li><strong>Architecture:</strong> Đọc <a class="cba-link cba-path-link" href="../docs/current_flow/CURRENT_FLOW_SUMMARY.md"><code class="cba-inline-code">CURRENT_FLOW_SUMMARY.md</code></a> → hiểu clean architecture.</li>
<li><strong>Phase Details:</strong> Đọc <a class="cba-link cba-path-link" href="../docs/current_flow/PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md"><code class="cba-inline-code">PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md</code></a> → hiểu foundation + modeling.</li>
<li><strong>Phase Details (cont.):</strong> Đọc <a class="cba-link cba-path-link" href="../docs/current_flow/PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md"><code class="cba-inline-code">PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md</code></a> → hiểu sweeps + final + analysis.</li>
<li><strong>Code Style:</strong> Đọc <a class="cba-link cba-path-link" href="../docs/rule_base/rule_code.md"><code class="cba-inline-code">docs/rule_base/rule_code.md</code></a> và <a class="cba-link cba-path-link" href="../docs/RULE_BASE/architecture_rule.md"><code class="cba-inline-code">architecture_rule.md</code></a>.</li>
<li><strong>Run:</strong> Open notebook <a class="cba-link cba-path-link" href="../notebook_course_work/CourseWork.ipynb"><code class="cba-inline-code">CourseWork.ipynb</code></a> với <a class="cba-link cba-path-link" href="../docs/link&discussion_to_result/NOTEBOOK_CELLS_WALKTHROUGH.md"><code class="cba-inline-code">NOTEBOOK_CELLS_WALKTHROUGH.md</code></a> bên cạnh.</li>
<li><strong>Debug:</strong> Search <a class="cba-link cba-path-link" href="../docs/analysis_error"><code class="cba-inline-code">docs/analysis_error/</code></a> cho issue tương tự.</li>
</ol>
<hr class="cba-rule"/>
<p><strong>Phiên bản:</strong> 06/09/2026 — sau lần refactor clean architecture lớn.
<strong>Maintainer:</strong> AI-assisted documentation</p>
</section></div>
<div class="cba-footer"><h3>Engineering Audit · COURSE_WORK</h3><p>Contract → data → models → sweeps → final governance → verified reporting. Historical provenance stays visible; current V2 presentation remains explicitly separated.</p><div class="cba-floaters"><span class="cba-floater">🐍</span><span class="cba-floater">🔥</span><span class="cba-floater">⚙️</span><span class="cba-floater">🧪</span><span class="cba-floater">✓</span></div><svg class="cba-wave" viewBox="0 0 1440 110" preserveAspectRatio="none" aria-hidden="true"><path d="M0,70 C260,120 420,18 700,65 C960,105 1180,20 1440,64 L1440,110 L0,110 Z" fill="#8fd3ff" opacity=".52"/><path d="M0,82 C280,36 520,118 820,72 C1110,28 1260,90 1440,55 L1440,110 L0,110 Z" fill="#f7c948" opacity=".48"/></svg></div>
</div>
