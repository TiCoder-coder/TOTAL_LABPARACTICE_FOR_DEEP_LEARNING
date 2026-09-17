<!-- Refactored visual Markdown. CSS/SVG only; no JavaScript. Source content and scientific lineage preserved. -->
<style>
:root{--cf-ink:#101828;--cf-muted:#475467;--cf-blue:#0b63ce;--cf-blue2:#2f80ed;--cf-cyan:#4cc9f0;--cf-yellow:#f5c84b;--cf-yellow-soft:#fff6c8;--cf-paper:#fff;--cf-soft:#f6faff;--cf-line:#d8e7f5;--cf-green:#10b981;--cf-green-soft:#eafbf4;--cf-purple:#7c3aed}
*{box-sizing:border-box}html{scroll-behavior:smooth}.cf-shell{max-width:1240px;margin:0 auto;padding:24px 20px 0;background:linear-gradient(180deg,#fff 0%,#f7fbff 43%,#fffdf4 100%);color:var(--cf-ink);font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;line-height:1.72}
.cf-hero{position:relative;isolation:isolate;overflow:hidden;border:1px solid #cfe2f7;border-radius:30px;padding:40px 38px 32px;background:radial-gradient(circle at 85% 12%,rgba(245,200,75,.34),transparent 23%),radial-gradient(circle at 9% 8%,rgba(76,201,240,.24),transparent 29%),linear-gradient(135deg,#fff 0%,#f3f9ff 58%,#fff9dc 100%);box-shadow:0 22px 65px rgba(15,81,145,.12)}.cf-hero:before,.cf-hero:after{content:"";position:absolute;border-radius:50%;z-index:-1}.cf-hero:before{width:245px;height:245px;right:-82px;bottom:-112px;border:1px solid rgba(11,99,206,.18);animation:cfPulse 5s ease-in-out infinite}.cf-hero:after{width:118px;height:118px;right:-16px;bottom:-38px;border:1px solid rgba(245,200,75,.62);animation:cfPulse 4s ease-in-out infinite reverse}.cf-kicker{font-size:12px;font-weight:850;letter-spacing:.18em;text-transform:uppercase;color:#0758b3}.cf-title{margin:7px 0 10px!important;color:#0b1220!important;font-size:clamp(32px,5vw,55px)!important;line-height:1.05!important;border:0!important}.cf-subtitle{max-width:930px;margin:0;color:#344054;font-size:16px}.cf-subtitle strong{color:#0b63ce}
.cf-badges{display:flex;flex-wrap:wrap;gap:10px;margin-top:22px}.cf-badge{display:inline-flex;align-items:center;gap:8px;padding:8px 12px;border:1px solid #d8e7f6;border-radius:999px;background:rgba(255,255,255,.9);font-size:13px;font-weight:750;box-shadow:0 6px 18px rgba(15,81,145,.08);transition:transform .22s ease,box-shadow .22s ease}.cf-badge:hover{transform:translateY(-3px);box-shadow:0 10px 24px rgba(15,81,145,.14)}.cf-badge svg{width:19px;height:19px;display:block}
.cf-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:22px}.cf-stat{padding:14px 15px;border-radius:16px;background:rgba(255,255,255,.84);border:1px solid #dce9f6;backdrop-filter:blur(4px)}.cf-stat-value{font-size:22px;font-weight:900;line-height:1.05;color:#0b63ce}.cf-stat:nth-child(3) .cf-stat-value{color:#7c3aed}.cf-stat:nth-child(4) .cf-stat-value{color:#087a57}.cf-stat-label{margin-top:6px;font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#667085}
.cf-hero-flow{display:grid;grid-template-columns:1fr auto 1fr auto 1fr auto 1fr auto 1fr;gap:8px;align-items:center;margin-top:22px}.cf-flow-node{padding:12px 10px;border-radius:14px;border:1px solid #d8e6f4;background:#fff;font-weight:800;text-align:center;box-shadow:0 5px 16px rgba(15,81,145,.06)}.cf-flow-node small{display:block;margin-top:2px;color:#667085;font-weight:650}.cf-flow-arrow{font-size:20px;color:#0b63ce;animation:cfArrow 1.8s ease-in-out infinite}
.cf-document{padding:10px 0 40px}.cf-section{position:relative;margin:20px 0;padding:25px 27px;background:rgba(255,255,255,.96);border:1px solid var(--cf-line);border-radius:21px;box-shadow:0 8px 28px rgba(15,81,145,.065);transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease}.cf-section:hover{transform:translateY(-2px);box-shadow:0 15px 36px rgba(15,81,145,.105);border-color:#bdd9f3}.cf-heading{scroll-margin-top:20px;color:#101828}.cf-h1{display:none}.cf-h2{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:0 0 15px!important;padding:0 0 12px;border-bottom:3px solid transparent;border-image:linear-gradient(90deg,var(--cf-blue),var(--cf-cyan),var(--cf-yellow),transparent) 1;font-size:27px!important;line-height:1.25!important}.cf-h3{margin:25px 0 10px!important;color:#0b5eb8!important;font-size:20px!important}.cf-h3:before{content:"◆";margin-right:8px;color:var(--cf-yellow);font-size:.72em}.cf-section-pill{display:inline-flex;align-items:center;padding:5px 8px;border-radius:8px;background:#eaf4ff;color:#0758b3;font-size:10px;font-weight:900;letter-spacing:.1em;white-space:nowrap}.cf-policy-card .cf-section-pill{background:#eafbf4;color:#087a57}.cf-gates-card .cf-section-pill,.cf-artifacts-card .cf-section-pill{background:#fff6cf;color:#7d5700}
.cf-toc-card{background:linear-gradient(135deg,#fbfdff,#f3f9ff 68%,#fffaf0)}.cf-toc-card:after{content:"Navigation Map";position:absolute;top:18px;right:22px;padding:5px 9px;border-radius:999px;background:#fff3b8;color:#7c5700;font-size:10px;font-weight:900;letter-spacing:.08em;text-transform:uppercase}.cf-toc-list>li{margin:9px 0}.cf-toc-list ul{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px 18px;margin-top:7px}.cf-status-card{background:radial-gradient(circle at 96% 8%,rgba(76,201,240,.18),transparent 28%),#fff}.cf-policy-card{background:radial-gradient(circle at 93% 6%,rgba(16,185,129,.13),transparent 28%),linear-gradient(135deg,#fff,#f5fffb 65%,#fffbea)}.cf-e2e-card{background:linear-gradient(135deg,#fff,#f5faff 68%,#fffaf0)}.cf-architecture-card,.cf-dataflow-card{background:linear-gradient(135deg,#fff,#f7fbff)}.cf-phases-card{background:radial-gradient(circle at 97% 4%,rgba(124,58,237,.08),transparent 24%),#fff}.cf-artifacts-card{background:radial-gradient(circle at 96% 5%,rgba(245,200,75,.18),transparent 28%),#fff}.cf-gates-card{background:linear-gradient(135deg,#fffdf4,#f8fbff)}.cf-docs-card{background:linear-gradient(135deg,#fff,#f7fbff 60%,#fff9df)}
.cf-nav-notice{display:grid;grid-template-columns:auto 1fr;gap:14px;align-items:start;margin:18px 0;padding:17px 18px;border-radius:16px;border:1px solid #cfe2f7;background:linear-gradient(90deg,#edf6ff,#fffdf0);box-shadow:0 7px 20px rgba(15,81,145,.06)}.cf-nav-notice-icon{width:38px;height:38px;display:grid;place-items:center;border-radius:11px;background:#0b63ce;color:white;font-size:20px;font-weight:900}.cf-nav-notice strong{color:#0758b3}.cf-nav-notice p{margin:3px 0 0;color:#475467}.cf-notebook-link{display:inline-flex!important;align-items:center;padding:2px 8px;border-radius:999px;background:#eaf4ff;border:1px solid #cfe2f7;text-decoration:none!important;font-size:.92em}.cf-notebook-link:hover{background:#ddecff;text-decoration:none!important}
.cf-policy-ribbon{display:grid;grid-template-columns:auto 1fr auto;gap:12px;align-items:center;margin:16px 0 20px;padding:15px 16px;border:1px solid #bcead9;border-radius:16px;background:linear-gradient(90deg,#ebfbf5,#f8fffc 60%,#fff8d8);box-shadow:0 8px 22px rgba(16,185,129,.08)}.cf-policy-icon{width:37px;height:37px;display:grid;place-items:center;border-radius:50%;background:#10b981;color:#fff;font-weight:900}.cf-policy-ribbon strong{display:block;color:#087a57;font-size:14px;letter-spacing:.04em}.cf-policy-ribbon span:not(.cf-policy-icon):not(.cf-policy-lock){display:block;color:#475467;font-size:13px}.cf-policy-lock{padding:5px 9px;border-radius:999px;background:#eafbf4;border:1px solid #bcead9;color:#087a57;font-size:10px;font-weight:900;letter-spacing:.08em}
.cf-note{margin:18px 0!important;padding:16px 18px!important;border-left:4px solid var(--cf-blue)!important;background:linear-gradient(90deg,#edf6ff,#fffdf2)!important;border-radius:0 13px 13px 0;color:#26364a}.cf-rule{border:0!important;height:1px!important;background:linear-gradient(90deg,transparent,#93c5fd,#facc15,transparent)!important;margin:28px 0!important}.cf-link{color:#075fbe!important;text-decoration:none!important;font-weight:650}.cf-link:hover{text-decoration:underline!important;text-decoration-thickness:1.5px!important}.cf-inline-code{padding:.14em .42em;border:1px solid #d9e7f4;border-radius:6px;background:#f3f8fc;color:#854d0e;font-size:.92em}.cf-code-blue{background:#eaf4ff!important;color:#0758b3!important;border-color:#c9def4!important}.cf-code-yellow{background:#fff6c9!important;color:#7b5600!important;border-color:#f0dc86!important}.cf-code-green{background:#eafbf4!important;color:#087a57!important;border-color:#bdebd9!important}.cf-code{overflow:auto;padding:17px 18px;border:1px solid #cfe0ef;border-radius:14px;background:#f7fbff;box-shadow:inset 4px 0 0 #4cc9f0;white-space:pre}.cf-code code{background:transparent!important;color:#111827!important}.cf-dataflow-card .cf-code{background:linear-gradient(135deg,#f7fbff,#fffdf5);box-shadow:inset 4px 0 0 #0b63ce}
.cf-table-wrap{overflow-x:auto;margin:18px 0;border:1px solid #d7e6f4;border-radius:14px;box-shadow:0 5px 18px rgba(15,81,145,.06)}.cf-table{width:100%;border-collapse:collapse;background:#fff;font-size:14px}.cf-table th{padding:11px 12px;background:linear-gradient(90deg,#eaf5ff,#fff6c9);color:#172033;text-align:left;border-bottom:1px solid #cbddeb}.cf-table td{padding:10px 12px;border-bottom:1px solid #edf2f7;vertical-align:top}.cf-table tr:last-child td{border-bottom:0}.cf-table tbody tr:hover{background:#f8fcff}ul,ol{padding-left:24px}li::marker{color:#0b63ce}strong{color:#111827}
.cf-footer{position:relative;overflow:hidden;margin:24px -20px 0;padding:66px 24px 94px;text-align:center;background:linear-gradient(180deg,#f8fcff,#e9f6ff 54%,#fff4bf);border-top:1px solid #cfe4f7}.cf-footer h3{margin:0;color:#101828;font-size:25px}.cf-footer p{margin:7px auto 0;max-width:760px;color:#475467}.cf-floaters{display:flex;justify-content:center;gap:18px;margin:23px 0 4px}.cf-floater{font-size:29px;display:inline-block;animation:cfFloat 3s ease-in-out infinite}.cf-floater:nth-child(2){animation-delay:.3s}.cf-floater:nth-child(3){animation-delay:.6s}.cf-floater:nth-child(4){animation-delay:.9s}.cf-floater:nth-child(5){animation-delay:1.2s}.cf-wave{position:absolute;left:-1%;right:-1%;bottom:-3px;width:102%;height:78px;opacity:.55}.cf-wave path:first-child{animation:cfWave 5s ease-in-out infinite alternate}.cf-wave path:last-child{animation:cfWave 6s ease-in-out infinite alternate-reverse}
@keyframes cfPulse{0%,100%{transform:scale(1);opacity:.62}50%{transform:scale(1.12);opacity:1}}@keyframes cfFloat{0%,100%{transform:translateY(0) rotate(0)}50%{transform:translateY(-11px) rotate(4deg)}}@keyframes cfArrow{0%,100%{transform:translateX(0);opacity:.65}50%{transform:translateX(4px);opacity:1}}@keyframes cfWave{from{transform:translateX(-8px)}to{transform:translateX(8px)}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important;scroll-behavior:auto!important}}@media(max-width:860px){.cf-stats{grid-template-columns:repeat(2,minmax(0,1fr))}.cf-hero-flow{grid-template-columns:1fr}.cf-flow-arrow{transform:rotate(90deg)}.cf-toc-list ul{grid-template-columns:1fr}.cf-policy-ribbon{grid-template-columns:auto 1fr}.cf-policy-lock{grid-column:2}.cf-section{padding:20px 18px}}@media(max-width:620px){.cf-shell{padding:12px 10px 0}.cf-hero{padding:26px 20px;border-radius:20px}.cf-stats{grid-template-columns:1fr 1fr}.cf-h2{font-size:23px!important}.cf-footer{margin-left:-10px;margin-right:-10px}}
</style>
<div class="cf-shell">
<div class="cf-hero">
<div class="cf-kicker">CourseWork · Current Architecture & Scientific Lineage</div>
<h1 class="cf-title">Current Flow Summary</h1>
<p class="cf-subtitle">Dashboard tổng quan cho <strong>Deep Learning Coursework</strong>: data pipeline, V1 scientific lineage, recovered sweeps, MODEL_IMPROVEMENT_V2 và final reporting policy.</p>
<div class="cf-badges"><span class="cf-badge"><svg viewBox="0 0 24 24" aria-hidden="true"><path fill="#3776AB" d="M12 2c-4 0-4 2-4 4v2h8v1H6c-2 0-4 1-4 4s2 4 4 4h2v-3c0-2 2-4 4-4h6c2 0 4-2 4-4s-2-4-4-4z"/><circle cx="10" cy="5" r="1" fill="#fff"/><path fill="#FFD43B" d="M12 22c4 0 4-2 4-4v-2H8v-1h10c2 0 4-1 4-4s-2-4-4-4h-2v3c0 2-2 4-4 4H6c-2 0-4 2-4 4s2 4 4 4z"/><circle cx="14" cy="19" r="1" fill="#fff"/></svg>Python</span><span class="cf-badge">🔥 PyTorch</span><span class="cf-badge">◫ Jupyter</span><span class="cf-badge">⚡ Transformer</span><span class="cf-badge">⌁ Time Series</span><span class="cf-badge">◈ Data Pipeline</span><span class="cf-badge">✓ Verified Artifacts</span></div>
<div class="cf-stats"><div class="cf-stat"><div class="cf-stat-value">59</div><div class="cf-stat-label">Coursework phases</div></div><div class="cf-stat"><div class="cf-stat-value">153</div><div class="cf-stat-label">Notebook cells</div></div><div class="cf-stat"><div class="cf-stat-value">E01–E20</div><div class="cf-stat-label">V2 improvement</div></div><div class="cf-stat"><div class="cf-stat-value">V2 FINAL</div><div class="cf-stat-label">Locked reporting</div></div></div>
<div class="cf-hero-flow"><div class="cf-flow-node">Raw Data<small>Acquire · Audit</small></div><div class="cf-flow-arrow">→</div><div class="cf-flow-node">V1 Core<small>Data · Models</small></div><div class="cf-flow-arrow">→</div><div class="cf-flow-node">Sweeps<small>S1–S19</small></div><div class="cf-flow-arrow">→</div><div class="cf-flow-node">V2 Improvement<small>E01–E20</small></div><div class="cf-flow-arrow">→</div><div class="cf-flow-node">Final Reporting<small>47–51 · 58–59</small></div></div>
</div>
</blockquote><div class="cf-nav-notice"><div aria-hidden="true" class="cf-nav-notice-icon">↗</div><div><strong>Notebook navigation note</strong><p><strong>Open Notebook ↗</strong> only opens <code>CourseWork.ipynb</code>. VS Code/Jupyter Markdown links do not reliably jump to an exact notebook cell. Use this summary and the cell-level walkthrough for precise phase/cell mapping.</p></div></div>
<section class="cf-section cf-toc-card"><h2 class="cf-heading cf-h2" id="mục-lục"><span class="cf-section-pill">NAVIGATION</span>Mục lục</h2>
<ul class="cf-toc-list">
<li><a class="cf-link" href="#trạng-thái-hiện-tại">Trạng thái hiện tại</a></li>
<li><a class="cf-link" href="#final-locked-v2-policy">Final locked V2 policy</a></li>
<li><a class="cf-link" href="#current-end-to-end-flow">Current end-to-end flow</a></li>
<li><a class="cf-link" href="#1-tổng-quan-project">1. Tổng quan project</a><ul>
<li><a class="cf-link" href="#11-mục-tiêu-coursework">Mục tiêu coursework</a></li>
<li><a class="cf-link" href="#12-tech-stack">Tech stack</a></li>
</ul>
</li>
<li><a class="cf-link" href="#2-kiến-trúc-source-code-sau-refactor">2. Kiến trúc source code</a></li>
<li><a class="cf-link" href="#3-tổng-quan-59-phases">3. Tổng quan 59 phases</a></li>
<li><a class="cf-link" href="#4-artifacts-layout">4. Artifacts layout</a></li>
<li><a class="cf-link" href="#5-data-flow-chi-tiết">5. Data flow chi tiết</a></li>
<li><a class="cf-link" href="#6-sign-off-lattice-phase-gates">6. Sign-off lattice</a></li>
<li><a class="cf-link" href="#7-scripts-layer">7. Scripts layer</a></li>
<li><a class="cf-link" href="#8-testing-layer">8. Testing layer</a></li>
<li><a class="cf-link" href="#9-tài-liệu-liên-quan">9. Tài liệu liên quan</a></li>
</ul>
</section><section class="cf-section cf-status-card"><h2 class="cf-heading cf-h2" id="trạng-thái-hiện-tại"><span class="cf-section-pill">STATUS</span>Trạng thái hiện tại</h2>
<div class="cf-table-wrap"><table class="cf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Trạng thái</th>
<th>Source</th>
</tr>
</thead>
<tbody>
<tr>
<td>Notebook</td>
<td>153 cells</td>
<td><a class="cf-link cf-notebook-link" href="../../notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Relative Markdown links do not reliably deep-link to an exact notebook cell.">Open Notebook ↗</a></td>
</tr>
<tr>
<td>Cell-level walkthrough</td>
<td>Current verified mapping</td>
<td><a class="cf-link" href="../link&amp;discussion_to_result/NOTEBOOK_CELLS_WALKTHROUGH.md">NOTEBOOK_CELLS_WALKTHROUGH.md</a></td>
</tr>
<tr>
<td>Phase 1–33</td>
<td>V1 current/historical flow</td>
<td><a class="cf-link" href="PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md">PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md</a></td>
</tr>
<tr>
<td>Phase 34–41</td>
<td><code class="cf-inline-code cf-code-blue">V1 RECOVERED</code>, <code class="cf-inline-code cf-code-blue">VALID_REUSABLE</code>, <code class="cf-inline-code cf-code-blue">RENDER_ONLY</code></td>
<td><a class="cf-link" href="PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md">PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md</a></td>
</tr>
<tr>
<td>Phase 42–46</td>
<td><code class="cf-inline-code cf-code-yellow">V1 HISTORICAL</code></td>
<td><a class="cf-link" href="PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md">PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md</a></td>
</tr>
<tr>
<td>Phase 47–51</td>
<td><code class="cf-inline-code cf-code-green">V2 FINAL</code></td>
<td><a class="cf-link" href="../../artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics_with_mape.json">Step 17 metrics</a></td>
</tr>
<tr>
<td>Phase 52–57</td>
<td><code class="cf-inline-code cf-code-yellow">V1 HISTORICAL ATTENTION</code></td>
<td>Historical attention artifacts</td>
</tr>
<tr>
<td>Phase 58–59</td>
<td><code class="cf-inline-code cf-code-green">V2 FINAL</code></td>
<td><a class="cf-link" href="../../artifacts/model_improvement_v2/final_reporting_analysis/manifest.json">Final reporting manifest</a></td>
</tr>
</tbody>
</table></div>
</section><section class="cf-section cf-policy-card"><h2 class="cf-heading cf-h2" id="final-locked-v2-policy"><span class="cf-section-pill">V2 FINAL</span>Final locked V2 policy</h2><div class="cf-policy-ribbon"><span class="cf-policy-icon">✓</span><div><strong>FINAL_LOCKED_POLICY</strong><span>Equal-weight ensemble · seeds 42 / 123 / 2026 · weights 1/3 each</span></div><span class="cf-policy-lock">LOCKED</span></div>
<p>Final policy: <code class="cf-inline-code cf-code-green">V2 Equal-weight Ensemble = FINAL_LOCKED_POLICY</code>; prediction = <code class="cf-inline-code">mean(E14-M1 seeds 42,123,2026)</code>, weights <code class="cf-inline-code">[1/3,1/3,1/3]</code>.</p>
<div class="cf-table-wrap"><table class="cf-table">
<thead>
<tr>
<th>Model / policy</th>
<th style="text-align:right">RMSE Wh</th>
<th style="text-align:right">MAE Wh</th>
<th style="text-align:right">MAPE %</th>
<th style="text-align:right">R²</th>
<th>Evidence</th>
</tr>
</thead>
<tbody>
<tr>
<td>V2 Equal-weight Ensemble</td>
<td style="text-align:right">61.608937</td>
<td style="text-align:right">25.897130</td>
<td style="text-align:right">21.519219</td>
<td style="text-align:right">0.540364</td>
<td><a class="cf-link" href="../../artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics_with_mape.json">Step 17 MAPE metrics</a></td>
</tr>
<tr>
<td>Persistence</td>
<td style="text-align:right">66.836915</td>
<td style="text-align:right">26.737589</td>
<td style="text-align:right">21.513353</td>
<td style="text-align:right">0.459047</td>
<td>Same frozen population</td>
</tr>
</tbody>
</table></div>
<p>Đây là <code class="cf-inline-code cf-code-green">POST_HOC_V2_BENCHMARK</code>, không phải unseen unbiased Test. Không chọn best seed và không post-Test retuning.</p>
</section><section class="cf-section cf-e2e-card"><h2 class="cf-heading cf-h2" id="current-end-to-end-flow"><span class="cf-section-pill">PIPELINE</span>Current end-to-end flow</h2>
<pre class="cf-code"><code class="language-text">Raw data → schema/temporal → chronological split → EDA
→ feature engineering → feature sets → scaling/windows
→ V1 models and sweeps → historical V1 final evidence
→ MODEL_IMPROVEMENT_V2 E01–E20 → Step 14 ensemble selection
→ Step 16 final refit/lock → Step 17 post-hoc benchmark
→ V2 reporting Phase 47–51/58–59
</code></pre>
<p>Phase 52–57 được giữ riêng như <code class="cf-inline-code cf-code-yellow">V1 HISTORICAL ATTENTION</code>; không dùng để kết luận attention behavior của final V2.</p>
<hr class="cf-rule"/>
</section><section class="cf-section cf-project-card"><h2 class="cf-heading cf-h2" id="1-tổng-quan-project"><span class="cf-section-pill">PROJECT</span>1. Tổng Quan Project</h2>
<h3 class="cf-heading cf-h3" id="11-mục-tiêu-coursework">1.1. Mục tiêu Coursework</h3>
<p>Project Deep Learning cho <strong>Multivariate Time-Series Regression</strong>, sử dụng dataset <strong>UCI Appliances Energy Prediction</strong>.</p>
<div class="cf-table-wrap"><table class="cf-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Mô tả</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Bài toán</strong></td>
<td>Dự đoán mức tiêu thụ điện năng (<code class="cf-inline-code">Appliances</code>, đo bằng <code class="cf-inline-code">Wh</code>) của các thiết bị gia dụng</td>
</tr>
<tr>
<td><strong>Input</strong></td>
<td>Cửa sổ lookback (<code class="cf-inline-code">lookback ∈ {36, 72, 144}</code> step tương đương 6h, 12h, 24h) chứa nhiều biến cảm biến</td>
</tr>
<tr>
<td><strong>Target</strong></td>
<td><code class="cf-inline-code">Appliances</code> (Wh), horizon = 1 step (10 phút)</td>
</tr>
<tr>
<td><strong>Sample definition</strong></td>
<td>$X[t-L+1:t] \rightarrow Appliances[t+1]$</td>
</tr>
<tr>
<td><strong>Task type</strong></td>
<td>Sequence-to-one regression</td>
</tr>
<tr>
<td><strong>Models so sánh</strong></td>
<td>Persistence baseline, LSTM, Transformer Encoder</td>
</tr>
<tr>
<td><strong>Evaluation</strong></td>
<td>MAE, RMSE, R² ở đơn vị Wh gốc</td>
</tr>
<tr>
<td><strong>Split</strong></td>
<td>Chronological Train / Validation / Test (70 / 15 / 15)</td>
</tr>
<tr>
<td><strong>Interpretability</strong></td>
<td>Phân tích Transformer attention maps</td>
</tr>
</tbody>
</table></div>
<h3 class="cf-heading cf-h3" id="12-tech-stack">1.2. Tech Stack</h3>
<div class="cf-table-wrap"><table class="cf-table">
<thead>
<tr>
<th>Component</th>
<th>Version / Setting</th>
</tr>
</thead>
<tbody>
<tr>
<td>Python</td>
<td>3.10.11</td>
</tr>
<tr>
<td>torch</td>
<td>2.13.0</td>
</tr>
<tr>
<td>numpy</td>
<td>2.2.6</td>
</tr>
<tr>
<td>pandas</td>
<td>2.3.3</td>
</tr>
<tr>
<td>scikit-learn</td>
<td>1.7.2</td>
</tr>
<tr>
<td>matplotlib</td>
<td>3.10.9</td>
</tr>
<tr>
<td>jupyter</td>
<td>1.1.1</td>
</tr>
<tr>
<td>Selected device</td>
<td>mps</td>
</tr>
<tr>
<td>Deterministic mode</td>
<td>D0</td>
</tr>
<tr>
<td>Default dtype</td>
<td>torch.float32</td>
</tr>
</tbody>
</table></div>
<hr class="cf-rule"/>
</section><section class="cf-section cf-architecture-card"><h2 class="cf-heading cf-h2" id="2-kiến-trúc-source-code-sau-refactor"><span class="cf-section-pill">ARCHITECTURE</span>2. Kiến Trúc Source Code (Sau Refactor)</h2>
<h3 class="cf-heading cf-h3" id="21-clean-architecture-phân-theo-tầng">2.1. Clean Architecture — Phân theo Tầng</h3>
<pre class="cf-code"><code>src/course_work/
│
├── # ============ CORE DOMAIN ============
├── models/              # Neural architectures (LSTM, Transformer, RevIN)
├── training/            # Training engine + losses
├── attention/           # Core attention utilities
├── baselines/           # Baseline models (persistence, LSTM_B0, Transformer_B0)
├── evaluation/          # Evaluation metrics
│
├── # ============ DATA PIPELINE ============
├── data/                # Data acquisition, schema, features, scaling, windows
│
├── # ============ CONFIGURATION &amp; GOVERNANCE ============
├── contracts/           # Coursework contract (single source of truth)
├── sweeps/              # Hyperparameter sweeps (S1-S19)
├── sanity/              # Forward pass sanity tests
├── diagnostics/         # Learning curve diagnostics
├── lstm_tuning/         # LSTM tuning helpers
├── rolling_origin/      # Rolling origin robustness evaluation
├── metric_addendum/     # MAPE metric addendum
├── experiments/         # Experiment tracking &amp; registry
├── model_improvement_v2/ # E01–E20, Step 16 final refit, Step 17 benchmark
│
├── # ============ INFRASTRUCTURE ============
├── utils/               # Artifacts, environment, reproducibility
├── verification/        # Phase 46 verification (NO-TRAIN)
├── scaling/             # Public re-export of data.scaling
│
├── # ============ EXECUTION (NO-TRAIN) ============
├── final_model_lock/    # Phase 45 - Final model lock
├── final_test_evaluation/  # Phase 47 - Final test evaluation
│
├── # ============ ANALYSIS PHASES ============
├── analysis/
│   ├── prediction_analysis/         # Phase 48
│   ├── residual_analysis/           # Phase 49
│   ├── error_regime_analysis/       # Phase 50
│   ├── worst_error_analysis/        # Phase 51
│   ├── attention_extraction/        # Phase 52
│   ├── attention_heatmaps/          # Phase 53
│   ├── last_query_attention/        # Phase 54
│   ├── head_comparison/             # Phase 55
│   ├── error_conditioned_attention/ # Phase 56
│   ├── seed_stability_attention/    # Phase 57
│   ├── final_tables/                # Phase 58
│   └── final_conclusions/           # Phase 59
│
├── # ============ REPORTING &amp; SCRIPTS ============
├── reporting/           # Dashboards, frozen evidence and V2 final reporting
└── scripts/             # Pipeline runner scripts (p39-p47)
</code></pre>
<h3 class="cf-heading cf-h3" id="22-nguyên-tắc-phân-tầng">2.2. Nguyên Tắc Phân Tầng</h3>
<div class="cf-table-wrap"><table class="cf-table">
<thead>
<tr>
<th>Tầng</th>
<th>Trách nhiệm</th>
<th>KHÔNG ĐƯỢC chứa</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>models/</strong></td>
<td>Kiến trúc neural network</td>
<td>Training loop, data loading</td>
</tr>
<tr>
<td><strong>training/</strong></td>
<td>Engine huấn luyện</td>
<td>Model architecture, business logic</td>
</tr>
<tr>
<td><strong>data/</strong></td>
<td>ETL, feature engineering, scaling</td>
<td>Model code, metrics</td>
</tr>
<tr>
<td><strong>contracts/</strong></td>
<td>Single source of truth cho cấu hình</td>
<td>Logic xử lý</td>
</tr>
<tr>
<td><strong>analysis/</strong></td>
<td>Phân tích post-training, đọc-only artifacts</td>
<td>Training, mutation</td>
</tr>
<tr>
<td><strong>execution/</strong></td>
<td>Final pipeline (NO-TRAIN)</td>
<td>Sweep code, exploratory</td>
</tr>
<tr>
<td><strong>scripts/</strong></td>
<td>Pipeline orchestrators</td>
<td>Business logic</td>
</tr>
<tr>
<td><strong>utils/</strong></td>
<td>I/O atomic, environment</td>
<td>Domain logic</td>
</tr>
</tbody>
</table></div>
<hr class="cf-rule"/>
</section><section class="cf-section cf-phases-card"><h2 class="cf-heading cf-h2" id="3-tổng-quan-59-phases"><span class="cf-section-pill">59 PHASES</span>3. Tổng Quan 59 Phases</h2>
<h3 class="cf-heading cf-h3" id="31-phases-theo-nhóm-và-lineage-hiện-tại">3.1. Phases Theo Nhóm và lineage hiện tại</h3>
<div class="cf-table-wrap"><table class="cf-table">
<thead>
<tr>
<th>Nhóm</th>
<th>Phases</th>
<th>Mục đích / lineage</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Foundation / Modeling</strong></td>
<td>1–22</td>
<td>V1 data, contracts, models, baselines và diagnostics</td>
</tr>
<tr>
<td><strong>Sweeps</strong></td>
<td>23–33</td>
<td>V1 S1–S11 verified/frozen evidence</td>
</tr>
<tr>
<td><strong>Recovered sweeps</strong></td>
<td>34–41</td>
<td><code class="cf-inline-code cf-code-blue">V1 RECOVERED</code>; <code class="cf-inline-code cf-code-blue">VALID_REUSABLE</code>, <code class="cf-inline-code cf-code-blue">RENDER_ONLY</code></td>
</tr>
<tr>
<td><strong>Historical final pipeline</strong></td>
<td>42–46</td>
<td><code class="cf-inline-code cf-code-yellow">V1 HISTORICAL</code></td>
</tr>
<tr>
<td><strong>Final benchmark/reporting</strong></td>
<td>47–51</td>
<td><code class="cf-inline-code cf-code-green">V2 FINAL</code></td>
</tr>
<tr>
<td><strong>Attention analyses</strong></td>
<td>52–57</td>
<td><code class="cf-inline-code cf-code-yellow">V1 HISTORICAL ATTENTION</code>; không phải V2 attention evidence</td>
</tr>
<tr>
<td><strong>Final summary/conclusions</strong></td>
<td>58–59</td>
<td><code class="cf-inline-code cf-code-green">V2 FINAL</code></td>
</tr>
<tr>
<td><strong>V2 overview</strong></td>
<td>Cells 151–152</td>
<td>E01–E20, Step 14, Step 16, Step 17 và closure</td>
</tr>
</tbody>
</table></div>
<h3 class="cf-heading cf-h3" id="32-phase-mapping-module">3.2. Phase Mapping → Module</h3>
<div class="cf-table-wrap"><table class="cf-table">
<thead>
<tr>
<th>Phase</th>
<th>Module Path</th>
<th>Trạng thái</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td><code class="cf-inline-code">course_work.contracts</code></td>
<td>✅</td>
</tr>
<tr>
<td>1</td>
<td><code class="cf-inline-code">course_work.utils.environment</code></td>
<td>✅</td>
</tr>
<tr>
<td>2</td>
<td><code class="cf-inline-code">course_work.data.acquisition</code></td>
<td>✅</td>
</tr>
<tr>
<td>3</td>
<td><code class="cf-inline-code">course_work.data.schema</code></td>
<td>✅</td>
</tr>
<tr>
<td>4</td>
<td><code class="cf-inline-code">course_work.data.temporal</code></td>
<td>✅</td>
</tr>
<tr>
<td>5</td>
<td><code class="cf-inline-code">course_work.data.splitting</code></td>
<td>✅</td>
</tr>
<tr>
<td>6</td>
<td><code class="cf-inline-code">course_work.data.eda</code></td>
<td>✅</td>
</tr>
<tr>
<td>7</td>
<td><code class="cf-inline-code">course_work.data.features</code></td>
<td>✅</td>
</tr>
<tr>
<td>8</td>
<td><code class="cf-inline-code">course_work.data.feature_sets</code></td>
<td>✅</td>
</tr>
<tr>
<td>9</td>
<td><code class="cf-inline-code">course_work.data.scaling</code></td>
<td>✅</td>
</tr>
<tr>
<td>10</td>
<td><code class="cf-inline-code">course_work.data.windows</code></td>
<td>✅</td>
</tr>
<tr>
<td>11</td>
<td><code class="cf-inline-code">course_work.data.datasets</code></td>
<td>✅</td>
</tr>
<tr>
<td>12</td>
<td><code class="cf-inline-code">course_work.evaluation.metrics</code></td>
<td>✅</td>
</tr>
<tr>
<td>13</td>
<td><code class="cf-inline-code">course_work.experiments.registry</code></td>
<td>✅</td>
</tr>
<tr>
<td>14</td>
<td><code class="cf-inline-code">course_work.baselines.persistence</code></td>
<td>✅</td>
</tr>
<tr>
<td>15</td>
<td><code class="cf-inline-code">course_work.models.lstm_regressor</code></td>
<td>✅</td>
</tr>
<tr>
<td>16</td>
<td><code class="cf-inline-code">course_work.models.transformer_regressor</code></td>
<td>✅</td>
</tr>
<tr>
<td>17</td>
<td><code class="cf-inline-code">course_work.attention.verification</code></td>
<td>✅</td>
</tr>
<tr>
<td>18</td>
<td><code class="cf-inline-code">course_work.sanity.forward_sanity</code></td>
<td>✅</td>
</tr>
<tr>
<td>19</td>
<td><code class="cf-inline-code">course_work.training.engine</code></td>
<td>✅</td>
</tr>
<tr>
<td>20</td>
<td><code class="cf-inline-code">course_work.baselines.lstm_baseline</code></td>
<td>✅</td>
</tr>
<tr>
<td>21</td>
<td><code class="cf-inline-code">course_work.baselines.transformer_b0</code></td>
<td>✅</td>
</tr>
<tr>
<td>22</td>
<td><code class="cf-inline-code">course_work.diagnostics.learning_diagnostics</code></td>
<td>✅</td>
</tr>
<tr>
<td>23–30</td>
<td><code class="cf-inline-code">course_work.sweeps.{feature_set,time_feature,...}</code></td>
<td>✅</td>
</tr>
<tr>
<td>31–40</td>
<td><code class="cf-inline-code">course_work.sweeps.{weight_decay,dropout,...,revin}</code></td>
<td>✅</td>
</tr>
<tr>
<td>41</td>
<td><code class="cf-inline-code">course_work.sweeps.boundary_protocol</code></td>
<td>✅</td>
</tr>
<tr>
<td>42</td>
<td><code class="cf-inline-code">course_work.experiments.phase_execution</code></td>
<td>✅</td>
</tr>
<tr>
<td>43</td>
<td><code class="cf-inline-code">course_work.lstm_tuning.stages</code></td>
<td>✅</td>
</tr>
<tr>
<td>44</td>
<td><code class="cf-inline-code">course_work.rolling_origin.folds</code></td>
<td>✅</td>
</tr>
<tr>
<td>45</td>
<td><code class="cf-inline-code">course_work.final_model_lock.candidate_lock</code></td>
<td>✅</td>
</tr>
<tr>
<td>46</td>
<td><code class="cf-inline-code">course_work.scripts.p46_three_seed_runs</code></td>
<td>✅</td>
</tr>
<tr>
<td>47</td>
<td><code class="cf-inline-code">course_work.final_test_evaluation.evaluation</code></td>
<td>✅</td>
</tr>
<tr>
<td>48</td>
<td><code class="cf-inline-code">course_work.analysis.prediction_analysis</code></td>
<td>✅</td>
</tr>
<tr>
<td>49</td>
<td><code class="cf-inline-code">course_work.analysis.residual_analysis</code></td>
<td>✅</td>
</tr>
<tr>
<td>50</td>
<td><code class="cf-inline-code">course_work.analysis.error_regime_analysis</code></td>
<td>✅</td>
</tr>
<tr>
<td>51</td>
<td><code class="cf-inline-code">course_work.analysis.worst_error_analysis</code></td>
<td>✅</td>
</tr>
<tr>
<td>52</td>
<td><code class="cf-inline-code">course_work.analysis.attention_extraction</code></td>
<td>✅</td>
</tr>
<tr>
<td>53</td>
<td><code class="cf-inline-code">course_work.analysis.attention_heatmaps</code></td>
<td>✅</td>
</tr>
<tr>
<td>54</td>
<td><code class="cf-inline-code">course_work.analysis.last_query_attention</code></td>
<td>✅</td>
</tr>
<tr>
<td>55</td>
<td><code class="cf-inline-code">course_work.analysis.head_comparison</code></td>
<td>✅</td>
</tr>
<tr>
<td>56</td>
<td><code class="cf-inline-code">course_work.analysis.error_conditioned_attention</code></td>
<td>✅</td>
</tr>
<tr>
<td>57</td>
<td><code class="cf-inline-code">course_work.analysis.seed_stability_attention</code></td>
<td>✅</td>
</tr>
<tr>
<td>58</td>
<td><code class="cf-inline-code">course_work.analysis.final_tables</code></td>
<td>✅</td>
</tr>
<tr>
<td>59</td>
<td><code class="cf-inline-code">course_work.analysis.final_conclusions</code></td>
<td>✅</td>
</tr>
</tbody>
</table></div>
<hr class="cf-rule"/>
</section><section class="cf-section cf-artifacts-card"><h2 class="cf-heading cf-h2" id="4-artifacts-layout"><span class="cf-section-pill">ARTIFACTS</span>4. Artifacts Layout</h2>
<pre class="cf-code"><code>artifacts/
├── contracts/                   # Phase 0
├── environment/                 # Phase 1
├── acquisition/                 # Phase 2
├── schema/                      # Phase 3
├── temporal/                    # Phase 4
├── eda/                         # Phase 6
├── features/                    # Phase 7
├── feature_sets/                # Phase 8
├── splits/                      # Phase 5
├── scaling/                     # Phase 9
├── scalers/                     # Phase 9 (joblib)
├── windows/                     # Phase 10
├── dataloaders/                 # Phase 11
├── metrics/                     # Phase 12
├── experiments/                 # Phase 13
├── baselines/persistence/       # Phase 14
├── runs/                        # All training runs (LSTM, Transformer, sweeps)
├── sweeps/S1..S19/              # Sweep results
├── lstm_tuning/                 # Phase 43
├── rolling_origin/              # Phase 44
├── final_model_lock/            # Phase 45 (NO-TRAIN)
├── three_seed_final_runs/       # Phase 46
├── final_test/                  # Phase 47 (NO-TRAIN)
├── prediction_analysis/         # Phase 48
├── residual_analysis/           # Phase 49
├── error_by_regime/             # Phase 50
├── worst_error_analysis/        # Phase 51
├── attention_extraction/        # Phase 52
├── attention_heatmaps/          # Phase 53
├── last_query_attention/        # Phase 54
├── head_comparison/             # Phase 55
├── error_conditioned_attention/ # Phase 56
├── seed_stability_attention/    # Phase 57
├── final_tables/                # Phase 58
└── final_conclusions/           # Phase 59
</code></pre>
<p>Mỗi phase có:</p>
<ul>
<li><code class="cf-inline-code">phase_N_signoff.json</code> — sign-off artifact (gate)</li>
<li><code class="cf-inline-code">_manifest.json</code> — inputs/outputs fingerprint</li>
<li><code class="cf-inline-code">_contract.json</code> — config lock</li>
<li><code class="cf-inline-code">_discrepancies.json</code> — issues tracker</li>
<li><code class="cf-inline-code">_report.md</code> — human-readable summary</li>
<li><code class="cf-inline-code">figures/</code> — plots</li>
<li><code class="cf-inline-code">tables/</code> — CSV/LaTeX/MD tables</li>
</ul>
<hr class="cf-rule"/>
</section><section class="cf-section cf-dataflow-card"><h2 class="cf-heading cf-h2" id="5-data-flow-chi-tiết"><span class="cf-section-pill">DATA FLOW</span>5. Data Flow Chi Tiết</h2>
<blockquote class="cf-note">
<p>Sơ đồ dưới đây giữ chi tiết subsystem của V1 workflow. Active final result và V2 flow được khóa tại phần đầu tài liệu; các block Phase 47–59 lịch sử không thay thế current V2 lineage.</p>
</blockquote>
<pre class="cf-code"><code>┌──────────────────────────────────────────────────────────────────────────────┐
│  RAW DATA (Phase 2)                                                          │
│  data/raw_data/energydata_complete.csv                                       │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ACQUISITION + SCHEMA (Phases 2-3)                                           │
│  src/course_work/data/acquisition.py, schema.py                              │
│  → artifacts/acquisition/, schema/                                           │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  TEMPORAL + SPLIT + EDA (Phases 4–6)                                                 │
│  → artifacts/temporal/, splits/, eda/                                                 │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  FEATURE ENGINEERING + FEATURE SETS (Phases 7–8)                             │
│  data/interim/uci_appliances_energy_prediction/                              │
│      energydata_feature_engineered_v1.csv                                    │
│  → artifacts/features/, feature_sets/                                        │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  SPLIT LINEAGE HANDOFF (Phase 5; reused downstream)                                                             │
│  → artifacts/splits/                                                         │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  SCALING + WINDOWS (Phases 9-10)                                             │
│  → artifacts/scaling/, scalers/, windows/                                    │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  DATALOADERS + METRICS (Phases 11-12)                                        │
│  → artifacts/dataloaders/, metrics/                                          │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  MODELS (Phases 13-22)                                                       │
│  - Persistence baseline (Phase 14)                                           │
│  - LSTM implementation (Phase 15)                                           │
│  - Transformer implementation (Phase 16)                                      │
│  - Attention verification (Phase 17)                                         │
│  - Forward sanity (Phase 18)                                                 │
│  - Training engine (Phase 19)                                                │
│  - Baseline runs (Phases 20-21)                                              │
│  - Learning diagnostics (Phase 22)                                          │
│  → artifacts/baselines/, runs/                                               │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  HYPERPARAMETER SWEEPS (Phases 23-41)                                        │
│  S1-S19 (feature set, lookback, d_model, layers, ffn, dropout, RevIN, ...)  │
│  → artifacts/sweeps/S1..S19/                                                 │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  CANDIDATE SYNTHESIS (Phase 42)                                              │
│  → artifacts/candidate_synthesis/                                             │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LSTM TUNING (Phase 43)                                                      │
│  → artifacts/lstm_tuning/                                                    │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ROLLING ORIGIN (Phase 44)                                                    │
│  → artifacts/rolling_origin/                                                 │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  FINAL MODEL LOCK (Phase 45 - NO-TRAIN)                                      │
│  src/course_work/final_model_lock/                                           │
│  → artifacts/final_model_lock/                                               │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  THREE-SEED FINAL RUNS (Phase 46)                                            │
│  src/course_work/scripts/p46_three_seed_runs.py                              │
│  → artifacts/three_seed_final_runs/                                          │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  HISTORICAL V1 FINAL TEST EVALUATION (Phase 47 provenance)                     │
│  src/course_work/final_test_evaluation/                                      │
│  → artifacts/final_test/                                                     │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  HISTORICAL V1 ANALYSIS (Phases 48-59 provenance)                                                     │
│  - Prediction analysis (48)                                                  │
│  - Residual analysis (49)                                                    │
│  - Error by regime (50)                                                      │
│  - Worst error (51)                                                          │
│  - Attention extraction → heatmaps → last query → head comparison →          │
│    error-conditioned → seed stability (52-57)                                 │
│  - Final tables (58)                                                         │
│  - Final conclusions (59)                                                    │
│  → artifacts/prediction_analysis/, residual_analysis/, ...                   │
└──────────────────────────────────────────────────────────────────────────────┘
</code></pre>
<hr class="cf-rule"/>
</section><section class="cf-section cf-gates-card"><h2 class="cf-heading cf-h2" id="6-sign-off-lattice-phase-gates"><span class="cf-section-pill">GATES</span>6. Sign-off Lattice (Phase Gates)</h2>
<p>Trong V1 historical workflow, mỗi phase tạo ra một <code class="cf-inline-code">phase_N_signoff.json</code> làm gate cho phase sau. Phase N+1 sẽ kiểm tra:</p>
<ul>
<li>Phase N's <code class="cf-inline-code">phase_N_signoff.json</code> tồn tại</li>
<li>Manifest checksums khớp</li>
<li>Required artifacts có mặt</li>
<li>Không có discrepancies mở</li>
</ul>
<p>Đây là cơ chế <strong>frozen contract</strong> đảm bảo upstream integrity trước khi chạy phase mới.</p>
<hr class="cf-rule"/>
</section><section class="cf-section cf-scripts-card"><h2 class="cf-heading cf-h2" id="7-scripts-layer"><span class="cf-section-pill">SCRIPTS</span>7. Scripts Layer</h2>
<pre class="cf-code"><code>src/course_work/scripts/
├── p39_cleanup_unauthorized_runs.py
├── p39_compliance_corrective.py
├── p39_finalize.py
├── p39_strict_best_gc0.py
├── p40_finalize.py
├── p40_prepare_revin.py
├── p40_strict_best_rn1.py
├── p42_candidate_synthesis.py
├── p43_disposable_harness.py
├── p43_dry_run.py
├── p43_lstm_tuning.py
├── p43_one_epoch_harness.py
├── p43_stage_simulation.py
├── p44_pretrain_gate.py
├── p44_quarantine_invalid_official.py
├── p44_rolling_origin.py
├── p45_final_model_lock.py
├── p45_pretrain_gate.py
├── p46_archive_historical_checkpoints.py
├── p46_final_dev_metric_smoke_test.py
├── p46_pre_train_schema_simulation.py
├── p46_pretrain_gate.py
├── p46_registration_preflight.py
├── p46_three_seed_runs.py
├── p47_final_test_evaluation.py
└── p47_pretest_gate.py
</code></pre>
<p>Mỗi script <code class="cf-inline-code">pXX_*.py</code> là <strong>pipeline runner</strong> chính thức cho phase XX, có thể gọi trực tiếp từ CLI hoặc notebook.</p>
<hr class="cf-rule"/>
</section><section class="cf-section cf-tests-card"><h2 class="cf-heading cf-h2" id="8-testing-layer"><span class="cf-section-pill">TESTS</span>8. Testing Layer</h2>
<pre class="cf-code"><code>tests/
├── contracts/         # Coursework contract tests
├── integration/       # Cross-phase integration tests
└── unit/              # Per-module unit tests (~80+ files)
</code></pre>
<hr class="cf-rule"/>
</section><section class="cf-section cf-docs-card"><h2 class="cf-heading cf-h2" id="9-tài-liệu-liên-quan"><span class="cf-section-pill">DOCS</span>9. Tài Liệu Liên Quan</h2>
<div class="cf-table-wrap"><table class="cf-table">
<thead>
<tr>
<th>File</th>
<th>Mô tả</th>
</tr>
</thead>
<tbody>
<tr>
<td><a class="cf-link" href="CURRENT_FLOW_SUMMARY.md">CURRENT_FLOW_SUMMARY.md</a></td>
<td>File này — tổng quan kiến trúc</td>
</tr>
<tr>
<td><a class="cf-link" href="PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md">PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md</a></td>
<td>Chi tiết phases 1–33 (foundation + modeling + sweeps)</td>
</tr>
<tr>
<td><a class="cf-link" href="PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md">PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md</a></td>
<td>Chi tiết phases 34–59 và lineage hiện tại</td>
</tr>
<tr>
<td><a class="cf-link" href="../../../README.md">README.md</a></td>
<td>Project root README</td>
</tr>
<tr>
<td><code class="cf-inline-code">docs/plan-doc/</code></td>
<td>Plan documents cho từng phase</td>
</tr>
<tr>
<td><a class="cf-link" href="../code_base_audit.md">code_base_audit.md</a></td>
<td>Code audit report</td>
</tr>
</tbody>
</table></div>
<hr class="cf-rule"/>
<p><strong>Phiên bản:</strong> 16/09/2026 — baseline chi tiết được giữ và đồng bộ với current verified flow.</p>
</section></div>
<div class="cf-footer"><h3>Current Flow · Verified Scientific Lineage</h3><p>Raw Data → V1 Foundation → Sweeps → Historical Evidence → MODEL_IMPROVEMENT_V2 → FINAL_LOCKED_POLICY → Final Reporting</p><div class="cf-floaters"><span class="cf-floater">🐍</span><span class="cf-floater">🔥</span><span class="cf-floater">⚡</span><span class="cf-floater">📊</span><span class="cf-floater">✓</span></div><svg class="cf-wave" viewBox="0 0 1200 120" preserveAspectRatio="none" aria-hidden="true"><path d="M0,60 C180,120 320,0 520,55 C720,110 900,10 1200,60 L1200,120 L0,120 Z" fill="#4cc9f0" opacity=".48"/><path d="M0,78 C220,25 380,115 610,66 C840,18 1010,100 1200,54 L1200,120 L0,120 Z" fill="#f5c84b" opacity=".46"/></svg></div>
</div>
