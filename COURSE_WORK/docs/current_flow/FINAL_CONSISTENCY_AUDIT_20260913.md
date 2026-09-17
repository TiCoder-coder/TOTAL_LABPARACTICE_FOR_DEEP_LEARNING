<!-- Refactored visual Markdown. CSS/SVG only; no JavaScript. Historical audit content preserved. -->
<style>
:root{--fa-ink:#101828;--fa-muted:#475467;--fa-blue:#0b63ce;--fa-cyan:#38bdf8;--fa-yellow:#f5c84b;--fa-paper:#fff;--fa-soft:#f6faff;--fa-line:#d7e6f4;--fa-green:#0f9f6e;--fa-green-soft:#eafbf4;--fa-amber:#b7791f;--fa-amber-soft:#fff7d6;--fa-red:#c2413b;--fa-red-soft:#fff1f0}
*{box-sizing:border-box}html{scroll-behavior:smooth}.fa-shell{max-width:1240px;margin:0 auto;padding:24px 20px 0;background:linear-gradient(180deg,#fff 0%,#f7fbff 44%,#fffdf3 100%);color:var(--fa-ink);font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;line-height:1.72}.fa-hero{position:relative;overflow:hidden;border:1px solid #cfe2f7;border-radius:30px;padding:40px 38px 32px;background:radial-gradient(circle at 88% 10%,rgba(245,200,75,.35),transparent 23%),radial-gradient(circle at 8% 8%,rgba(56,189,248,.22),transparent 28%),linear-gradient(135deg,#fff 0%,#f2f9ff 60%,#fff8d8 100%);box-shadow:0 22px 65px rgba(15,81,145,.12)}.fa-hero:before,.fa-hero:after{content:"";position:absolute;border-radius:50%;pointer-events:none}.fa-hero:before{width:250px;height:250px;right:-92px;bottom:-124px;border:1px solid rgba(11,99,206,.18);animation:faPulse 5s ease-in-out infinite}.fa-hero:after{width:120px;height:120px;right:-18px;bottom:-40px;border:1px solid rgba(245,200,75,.62);animation:faPulse 4s ease-in-out infinite reverse}.fa-kicker{font-size:12px;font-weight:900;letter-spacing:.18em;text-transform:uppercase;color:#0758b3}.fa-title{margin:7px 0 10px;color:#0b1220;font-size:clamp(32px,5vw,55px);line-height:1.05}.fa-subtitle{max-width:900px;color:#344054}.fa-badges{display:flex;flex-wrap:wrap;gap:10px;margin-top:21px}.fa-badge{display:inline-flex;align-items:center;gap:8px;padding:8px 12px;border:1px solid #d8e7f6;border-radius:999px;background:rgba(255,255,255,.9);font-size:13px;font-weight:800;box-shadow:0 6px 18px rgba(15,81,145,.08);transition:.2s}.fa-badge:hover{transform:translateY(-3px);box-shadow:0 10px 24px rgba(15,81,145,.14)}.fa-badge svg{width:19px;height:19px}.fa-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:22px}.fa-stat{padding:14px 15px;border-radius:16px;background:rgba(255,255,255,.84);border:1px solid #dce9f6}.fa-stat-value{font-size:21px;font-weight:900;color:#0b63ce}.fa-stat:nth-child(3) .fa-stat-value{color:#087a57}.fa-stat:nth-child(4) .fa-stat-value{color:#8a6200}.fa-stat-label{margin-top:5px;font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#667085}.fa-warning{display:grid;grid-template-columns:auto 1fr;gap:14px;margin:18px 0 0;padding:17px 18px;border:1px solid #f2d16d;border-radius:17px;background:linear-gradient(90deg,#fff8d8,#fffdf4 58%,#eef7ff);box-shadow:0 8px 22px rgba(145,105,15,.08)}.fa-warning-icon{width:40px;height:40px;display:grid;place-items:center;border-radius:12px;background:#f5c84b;color:#322500;font-weight:1000}.fa-warning strong{display:block;color:#7b5600}.fa-warning p{margin:2px 0 0;color:#475467}.fa-nav{display:flex;flex-wrap:wrap;gap:8px;margin-top:17px}.fa-nav a{padding:7px 10px;border:1px solid #cfe2f7;border-radius:999px;background:#fff;color:#0758b3;text-decoration:none;font-size:12px;font-weight:800}.fa-nav a:hover{background:#eaf4ff}.fa-document{padding:10px 0 40px}.fa-heading{scroll-margin-top:20px;color:#101828}.fa-h1{display:none}.fa-section{position:relative;margin:20px 0;padding:25px 27px;background:rgba(255,255,255,.97);border:1px solid var(--fa-line);border-radius:21px;box-shadow:0 8px 28px rgba(15,81,145,.065);transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease}.fa-section:hover{transform:translateY(-2px);box-shadow:0 15px 36px rgba(15,81,145,.105);border-color:#bdd9f3}.fa-h2{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:0 0 15px!important;padding:0 0 12px;border-bottom:3px solid transparent;border-image:linear-gradient(90deg,var(--fa-blue),var(--fa-cyan),var(--fa-yellow),transparent) 1;font-size:27px!important;line-height:1.25!important}.fa-h3{margin:25px 0 10px!important;color:#0b5eb8!important}.fa-section-pill{display:inline-flex;padding:5px 8px;border-radius:8px;background:#eaf4ff;color:#0758b3;font-size:10px;font-weight:900;letter-spacing:.1em;white-space:nowrap}.fa-checksum-card{background:radial-gradient(circle at 96% 5%,rgba(15,159,110,.10),transparent 26%),#fff}.fa-debt-card{background:radial-gradient(circle at 96% 5%,rgba(245,200,75,.20),transparent 28%),#fff}.fa-validation-card,.fa-conclusion-card{background:linear-gradient(135deg,#fff,#f5fffb 64%,#fffbea)}.fa-debt-card .fa-section-pill{background:#fff3bd;color:#7b5600}.fa-validation-card .fa-section-pill,.fa-conclusion-card .fa-section-pill,.fa-checksum-card .fa-section-pill{background:#eafbf4;color:#087a57}.fa-note{margin:18px 0!important;padding:16px 18px!important;border-left:4px solid var(--fa-yellow)!important;background:linear-gradient(90deg,#fff7d3,#f7fbff)!important;border-radius:0 13px 13px 0;color:#26364a}.fa-rule{border:0!important;height:1px!important;background:linear-gradient(90deg,transparent,#93c5fd,#facc15,transparent)!important;margin:28px 0!important}.fa-link{color:#075fbe!important;text-decoration:none!important;font-weight:650}.fa-link:hover{text-decoration:underline!important}.fa-inline-code{padding:.14em .42em;border:1px solid #d9e7f4;border-radius:6px;background:#f3f8fc;color:#854d0e;font-size:.92em}.fa-code-blue{background:#eaf4ff!important;color:#0758b3!important;border-color:#c9def4!important}.fa-code-green{background:#eafbf4!important;color:#087a57!important;border-color:#bdebd9!important}.fa-code-yellow{background:#fff6c9!important;color:#7b5600!important;border-color:#f0dc86!important}.fa-code{overflow:auto;padding:17px 18px;border:1px solid #cfe0ef;border-radius:14px;background:#f7fbff;box-shadow:inset 4px 0 0 #4cc9f0;white-space:pre}.fa-code code{background:transparent!important;color:#111827!important}.fa-table-wrap{overflow-x:auto;margin:18px 0;border:1px solid #d7e6f4;border-radius:14px;box-shadow:0 5px 18px rgba(15,81,145,.06)}.fa-table{width:100%;border-collapse:collapse;background:#fff;font-size:14px}.fa-table th{padding:11px 12px;background:linear-gradient(90deg,#eaf5ff,#fff6c9);color:#172033;text-align:left;border-bottom:1px solid #cbddeb}.fa-table td{padding:10px 12px;border-bottom:1px solid #edf2f7;vertical-align:top}.fa-table tr:last-child td{border-bottom:0}.fa-table tbody tr:hover{background:#f8fcff}ul,ol{padding-left:24px}li::marker{color:#0b63ce}strong{color:#111827}.fa-conclusion-strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:18px 0}.fa-conclusion-item{padding:13px;border:1px solid #cfe8dd;border-radius:14px;background:#f4fcf8;text-align:center}.fa-conclusion-item b{display:block;color:#087a57;font-size:13px}.fa-conclusion-item span{font-size:11px;color:#667085}.fa-footer{position:relative;overflow:hidden;margin:24px -20px 0;padding:66px 24px 94px;text-align:center;background:linear-gradient(180deg,#f8fcff,#e9f6ff 54%,#fff4bf);border-top:1px solid #cfe4f7}.fa-footer h3{margin:0;color:#101828;font-size:25px}.fa-footer p{margin:7px auto 0;max-width:760px;color:#475467}.fa-floaters{display:flex;justify-content:center;gap:18px;margin:23px 0 4px}.fa-floater{font-size:29px;display:inline-block;animation:faFloat 3s ease-in-out infinite}.fa-floater:nth-child(2){animation-delay:.3s}.fa-floater:nth-child(3){animation-delay:.6s}.fa-floater:nth-child(4){animation-delay:.9s}.fa-floater:nth-child(5){animation-delay:1.2s}.fa-wave{position:absolute;left:-1%;right:-1%;bottom:-3px;width:102%;height:78px;opacity:.58}.fa-wave path:first-child{animation:faWave 8s ease-in-out infinite}.fa-wave path:last-child{animation:faWave 6s ease-in-out infinite reverse}@keyframes faPulse{0%,100%{transform:scale(1);opacity:.55}50%{transform:scale(1.08);opacity:.95}}@keyframes faFloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-9px)}}@keyframes faWave{0%,100%{transform:translateX(0)}50%{transform:translateX(18px)}}@media(max-width:840px){.fa-stats,.fa-conclusion-strip{grid-template-columns:repeat(2,1fr)}.fa-hero{padding:30px 23px}.fa-section{padding:21px 18px}}@media(max-width:520px){.fa-stats,.fa-conclusion-strip{grid-template-columns:1fr}.fa-title{font-size:34px}.fa-shell{padding-left:12px;padding-right:12px}.fa-footer{margin-left:-12px;margin-right:-12px}}@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important;scroll-behavior:auto!important}}
</style>
<div class="fa-shell">
<div class="fa-hero">
<div class="fa-kicker">CourseWork · Evidence Integrity · Audit Snapshot</div>
<div class="fa-title">Final Consistency Audit</div>
<div class="fa-subtitle">Historical read-only audit snapshot dated <strong>2026-09-13</strong>. Scientific evidence is preserved; current presentation mapping is documented separately.</div>
<div class="fa-badges">
<span class="fa-badge"><svg viewBox="0 0 24 24" aria-hidden="true"><path fill="#3776AB" d="M12 2.2c-4.8 0-4.5 2.1-4.5 2.1l.01 2.2h4.58v.66H5.69S2.6 6.8 2.6 11.68s2.7 4.7 2.7 4.7h1.61v-2.26s-.09-2.7 2.66-2.7h4.55s2.55.04 2.55-2.47V4.74S17.05 2.2 12 2.2z"/><circle cx="9.48" cy="4.82" r=".78" fill="#fff"/><path fill="#FFD43B" d="M12 21.8c4.8 0 4.5-2.1 4.5-2.1l-.01-2.2h-4.58v-.66h6.4s3.09.36 3.09-4.52-2.7-4.7-2.7-4.7h-1.61v2.26s.09 2.7-2.66 2.7H9.88s-2.55-.04-2.55 2.47v4.21S6.95 21.8 12 21.8z"/><circle cx="14.52" cy="19.18" r=".78" fill="#fff"/></svg> Python</span><span class="fa-badge">✓ SHA-256</span><span class="fa-badge">▦ Artifacts</span><span class="fa-badge">⌁ Phase Gates</span><span class="fa-badge">◈ Read-only Audit</span><span class="fa-badge">Git ✓</span>
</div>
<div class="fa-stats"><div class="fa-stat"><div class="fa-stat-value">1–59</div><div class="fa-stat-label">Phases audited</div></div><div class="fa-stat"><div class="fa-stat-value">163 / 1</div><div class="fa-stat-label">Passed / Skipped</div></div><div class="fa-stat"><div class="fa-stat-value">389 / 0</div><div class="fa-stat-label">Links checked / Broken</div></div><div class="fa-stat"><div class="fa-stat-value">NO</div><div class="fa-stat-label">Training · Inference · Test access</div></div></div>
<div class="fa-warning"><div class="fa-warning-icon">!</div><div><strong>HISTORICAL AUDIT SNAPSHOT — SUPERSEDED FOR CURRENT PRESENTATION MAPPING</strong><p>This file preserves the 2026-09-13 audit conclusions. Use the current flow documents for the latest presentation mapping; do not reinterpret historical scientific artifacts.</p></div></div>
<div class="fa-nav"><a href="#pham-vi-va-nguyen-tac">Scope</a><a href="#trang-thai-phase-1-59">Phase status</a><a href="#exact-checksum-artifact-restoration">Checksum restore</a><a href="#phase-36-41-notebook-refresh-audit">Notebook audit</a><a href="#reporting-debt-con-giu-nguyen">Reporting debt</a><a href="#validation-cuoi">Validation</a><a href="#ket-luan">Conclusion</a></div>
</div>
<div class="fa-document"><h1 class="fa-heading fa-h1" id="final-consistency-audit-2026-09-13">Final consistency audit — 2026-09-13</h1>
<blockquote class="fa-note">
<p><strong>HISTORICAL AUDIT SNAPSHOT — SUPERSEDED FOR CURRENT PRESENTATION MAPPING</strong></p>
<p>File này giữ nguyên kết luận audit tại ngày <code class="fa-inline-code">2026-09-13</code>; không phải tài liệu mô tả presentation mới nhất. Trạng thái hiện tại phải đọc tại <a class="fa-link" href="CURRENT_FLOW_SUMMARY.md">CURRENT_FLOW_SUMMARY.md</a>, <a class="fa-link" href="PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md">PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md</a> và <a class="fa-link" href="../link&amp;discussion_to_result/NOTEBOOK_CELLS_WALKTHROUGH.md">NOTEBOOK_CELLS_WALKTHROUGH.md</a>. Sau snapshot này, Phase 47–51 và 58–59 được trình bày theo <code class="fa-inline-code fa-code-blue">V2 FINAL</code>; Phase 52–57 vẫn là <code class="fa-inline-code">V1 HISTORICAL ATTENTION</code>. Historical scientific artifacts và các kết luận audit bên dưới không bị viết lại.</p>
</blockquote>
<section class="fa-section fa-scope-card"><h2 class="fa-heading fa-h2" id="pham-vi-va-nguyen-tac"><span class="fa-section-pill">SCOPE</span>Phạm vi và nguyên tắc</h2>
<p>Audit này là read-only đối với scientific evidence. Không training, không
inference, không đọc Test source và không thay đổi checkpoint, scaler, metric,
prediction hay historical signoff. Các artifact được restore chỉ được chép vào
canonical path còn thiếu sau khi SHA-256 của source khớp checksum đã khóa.</p>
</section><section class="fa-section fa-phase-card"><h2 class="fa-heading fa-h2" id="trang-thai-phase-1-59"><span class="fa-section-pill">PHASE STATUS</span>Trạng thái Phase 1–59</h2>
<div class="fa-table-wrap"><table class="fa-table">
<thead>
<tr>
<th style="text-align:right">Phase</th>
<th>State</th>
<th>Scientific core</th>
<th>Signoff/artifacts</th>
<th>Processing log / presentation</th>
<th>Classification</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right">1</td>
<td><code class="fa-inline-code fa-code-green">PASS</code></td>
<td>Hợp lệ</td>
<td>Hợp lệ</td>
<td>Renderable</td>
<td><code class="fa-inline-code fa-code-green">STABLE</code></td>
</tr>
<tr>
<td style="text-align:right">2</td>
<td><code class="fa-inline-code fa-code-green">PASS</code></td>
<td>Hợp lệ sau exact-SHA restore</td>
<td>6/6 artifact đã restore đúng checksum</td>
<td>Renderable</td>
<td><code class="fa-inline-code fa-code-green">STABLE</code></td>
</tr>
<tr>
<td style="text-align:right">3</td>
<td><code class="fa-inline-code fa-code-yellow">PASS_WITH_WARNING</code></td>
<td>Hợp lệ</td>
<td>Hợp lệ</td>
<td>Renderable</td>
<td><code class="fa-inline-code fa-code-yellow">STABLE_WITH_REPORTING_DEBT</code></td>
</tr>
<tr>
<td style="text-align:right">4–6</td>
<td><code class="fa-inline-code fa-code-green">PASS</code></td>
<td>Hợp lệ</td>
<td>Hợp lệ</td>
<td>Renderable</td>
<td><code class="fa-inline-code fa-code-green">STABLE</code></td>
</tr>
<tr>
<td style="text-align:right">7</td>
<td><code class="fa-inline-code fa-code-green">PASS</code></td>
<td>Hợp lệ sau exact-SHA restore</td>
<td>Feature-engineered CSV đã restore đúng checksum</td>
<td>Renderable</td>
<td><code class="fa-inline-code fa-code-green">STABLE</code></td>
</tr>
<tr>
<td style="text-align:right">8–10</td>
<td><code class="fa-inline-code fa-code-green">PASS</code></td>
<td>Hợp lệ</td>
<td>Hợp lệ</td>
<td>Renderable</td>
<td><code class="fa-inline-code fa-code-green">STABLE</code></td>
</tr>
<tr>
<td style="text-align:right">11</td>
<td><code class="fa-inline-code fa-code-green">PASS</code></td>
<td>Hợp lệ sau exact-SHA restore</td>
<td>3/3 registry/audit CSV đã restore đúng checksum</td>
<td>Renderable</td>
<td><code class="fa-inline-code fa-code-green">STABLE</code></td>
</tr>
<tr>
<td style="text-align:right">12–13</td>
<td><code class="fa-inline-code fa-code-green">PASS</code></td>
<td>Hợp lệ</td>
<td>Hợp lệ</td>
<td>Renderable</td>
<td><code class="fa-inline-code fa-code-green">STABLE</code></td>
</tr>
<tr>
<td style="text-align:right">14</td>
<td><code class="fa-inline-code fa-code-green">PASS</code></td>
<td>Hợp lệ sau exact-SHA restore</td>
<td>Validation prediction CSV đã restore đúng checksum</td>
<td>Renderable</td>
<td><code class="fa-inline-code fa-code-green">STABLE</code></td>
</tr>
<tr>
<td style="text-align:right">15–22</td>
<td><code class="fa-inline-code fa-code-green">PASS</code></td>
<td>Hợp lệ</td>
<td>Hợp lệ</td>
<td>Renderable</td>
<td><code class="fa-inline-code fa-code-green">STABLE</code></td>
</tr>
<tr>
<td style="text-align:right">23–30</td>
<td><code class="fa-inline-code fa-code-green">VALID_REUSABLE</code></td>
<td>Hợp lệ</td>
<td>Required artifact count đầy đủ</td>
<td>Live action <code class="fa-inline-code fa-code-green">RENDER_ONLY</code>; saved log còn nhãn historical <code class="fa-inline-code fa-code-yellow">BLOCKED</code></td>
<td><code class="fa-inline-code fa-code-yellow">STABLE_WITH_REPORTING_DEBT</code></td>
</tr>
<tr>
<td style="text-align:right">31–33</td>
<td><code class="fa-inline-code fa-code-green">VALID_REUSABLE</code></td>
<td>Hợp lệ</td>
<td>Required artifact count đầy đủ</td>
<td><code class="fa-inline-code fa-code-green">RENDER_ONLY</code></td>
<td><code class="fa-inline-code fa-code-green">STABLE</code></td>
</tr>
<tr>
<td style="text-align:right">34–36</td>
<td><code class="fa-inline-code fa-code-green">VALID_REUSABLE</code></td>
<td>Canonical run/output và recovered signoff đúng checksum</td>
<td>Hợp lệ</td>
<td><code class="fa-inline-code fa-code-green">RENDER_ONLY</code>; log hợp lệ</td>
<td><code class="fa-inline-code fa-code-green">STABLE</code></td>
</tr>
<tr>
<td style="text-align:right">37</td>
<td><code class="fa-inline-code fa-code-green">VALID_REUSABLE</code></td>
<td>Canonical <code class="fa-inline-code">RUN_TR_S15_0024_9420CDD7</code> verified</td>
<td>Exact legacy signoff verified</td>
<td><code class="fa-inline-code fa-code-green">RENDER_ONLY</code></td>
<td><code class="fa-inline-code fa-code-yellow">STABLE_WITH_REPORTING_DEBT</code></td>
</tr>
<tr>
<td style="text-align:right">38</td>
<td><code class="fa-inline-code fa-code-green">VALID_REUSABLE</code></td>
<td>Canonical <code class="fa-inline-code">RUN_TR_S16_0025_49060872</code> verified</td>
<td>Exact legacy signoff verified</td>
<td><code class="fa-inline-code fa-code-green">RENDER_ONLY</code></td>
<td><code class="fa-inline-code fa-code-yellow">STABLE_WITH_REPORTING_DEBT</code></td>
</tr>
<tr>
<td style="text-align:right">39</td>
<td><code class="fa-inline-code fa-code-green">VALID_REUSABLE</code></td>
<td>Canonical <code class="fa-inline-code">RUN_TR_S17_0029_082F7FF5</code> verified</td>
<td>Exact legacy signoff verified</td>
<td><code class="fa-inline-code fa-code-green">RENDER_ONLY</code></td>
<td><code class="fa-inline-code fa-code-yellow">STABLE_WITH_REPORTING_DEBT</code></td>
</tr>
<tr>
<td style="text-align:right">40</td>
<td><code class="fa-inline-code fa-code-green">VALID_REUSABLE</code></td>
<td>Canonical <code class="fa-inline-code">RUN_TR_S18_0031_A711A9B8</code> verified</td>
<td>Exact legacy signoff verified</td>
<td><code class="fa-inline-code fa-code-green">RENDER_ONLY</code></td>
<td><code class="fa-inline-code fa-code-yellow">STABLE_WITH_REPORTING_DEBT</code></td>
</tr>
<tr>
<td style="text-align:right">41</td>
<td><code class="fa-inline-code fa-code-green">VALID_REUSABLE</code></td>
<td>Canonical <code class="fa-inline-code">RUN_TR_S19_0034_CF8C1FE8</code> verified</td>
<td>Exact legacy signoff verified</td>
<td><code class="fa-inline-code fa-code-green">RENDER_ONLY</code></td>
<td><code class="fa-inline-code fa-code-yellow">STABLE_WITH_REPORTING_DEBT</code></td>
</tr>
<tr>
<td style="text-align:right">42</td>
<td><code class="fa-inline-code fa-code-green">PASS</code></td>
<td>Hợp lệ</td>
<td>Hợp lệ</td>
<td>Renderable</td>
<td><code class="fa-inline-code fa-code-green">STABLE</code></td>
</tr>
<tr>
<td style="text-align:right">43–44</td>
<td>Verified result renderable</td>
<td>Signoff hiện tại đủ cho notebook</td>
<td>Core signoff hợp lệ</td>
<td>Generic processing-log rebuild thiếu reporting CSV</td>
<td><code class="fa-inline-code fa-code-yellow">STABLE_WITH_REPORTING_DEBT</code></td>
</tr>
<tr>
<td style="text-align:right">45–50</td>
<td>Verified result renderable</td>
<td>Hợp lệ</td>
<td>Current machine-readable source tồn tại</td>
<td>3-table read-only renderer</td>
<td><code class="fa-inline-code fa-code-green">STABLE</code></td>
</tr>
<tr>
<td style="text-align:right">51</td>
<td>Verified result renderable</td>
<td>Hợp lệ</td>
<td>Summary/signoff/figures hiện có</td>
<td>2 historical reporting CSV còn thiếu</td>
<td><code class="fa-inline-code fa-code-yellow">STABLE_WITH_REPORTING_DEBT</code></td>
</tr>
<tr>
<td style="text-align:right">52–53</td>
<td>Verified result renderable</td>
<td>Hợp lệ</td>
<td>Current machine-readable source tồn tại</td>
<td>3-table read-only renderer</td>
<td><code class="fa-inline-code fa-code-green">STABLE</code></td>
</tr>
<tr>
<td style="text-align:right">54–57</td>
<td>Verified result renderable</td>
<td>Hợp lệ</td>
<td>Summary/signoff hiện có; historical dashboard snapshot checksum-verified</td>
<td>Một số intermediate CSV vẫn là debt</td>
<td><code class="fa-inline-code fa-code-yellow">STABLE_WITH_REPORTING_DEBT</code></td>
</tr>
<tr>
<td style="text-align:right">58–59</td>
<td>Verified result renderable</td>
<td>Hợp lệ</td>
<td>Current machine-readable source tồn tại</td>
<td>3-table read-only renderer</td>
<td><code class="fa-inline-code fa-code-green">STABLE</code></td>
</tr>
</tbody>
</table></div>
</section><section class="fa-section fa-checksum-card"><h2 class="fa-heading fa-h2" id="exact-checksum-artifact-restoration"><span class="fa-section-pill">CHECKSUM</span>Exact-checksum artifact restoration</h2>
<p>Manifest đầy đủ: <a class="fa-link" href="../../artifacts/_reconstruction_history/final_consistency_exact_artifact_restore_20260913T061808Z.json">final_consistency_exact_artifact_restore_20260913T061808Z.json</a>.</p>
<div class="fa-table-wrap"><table class="fa-table">
<thead>
<tr>
<th style="text-align:right">Phase</th>
<th>Artifact group</th>
<th style="text-align:right">Count</th>
<th>Checksum result</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right">2</td>
<td>Raw-data source metadata, manifest, checksum ledger, ZIP và variable metadata</td>
<td style="text-align:right">6</td>
<td><code class="fa-inline-code fa-code-green">EXACT_SHA_MATCH</code></td>
</tr>
<tr>
<td style="text-align:right">7</td>
<td><code class="fa-inline-code">energydata_feature_engineered_v1.csv</code></td>
<td style="text-align:right">1</td>
<td><code class="fa-inline-code fa-code-green">EXACT_SHA_MATCH</code></td>
</tr>
<tr>
<td style="text-align:right">11</td>
<td>Dataset/DataLoader registries và shuffle audit</td>
<td style="text-align:right">3</td>
<td><code class="fa-inline-code fa-code-green">EXACT_SHA_MATCH</code></td>
</tr>
<tr>
<td style="text-align:right">14</td>
<td><code class="fa-inline-code">persistence_validation_predictions.csv</code></td>
<td style="text-align:right">1</td>
<td><code class="fa-inline-code fa-code-green">EXACT_SHA_MATCH</code></td>
</tr>
</tbody>
</table></div>
<p>Mọi destination đều absent trước copy. Không canonical file nào bị overwrite.</p>
</section><section class="fa-section fa-refresh-card"><h2 class="fa-heading fa-h2" id="phase-36-41-notebook-refresh-audit"><span class="fa-section-pill">NOTEBOOK</span>Phase 36–41 notebook refresh audit</h2>
<p>Saved HTML output của từng Phase 36–41 bằng byte với output tạo in-memory bởi
<code class="fa-inline-code">render_frozen_phase_evidence()</code> ở current tree. Sáu output đều chứa
<code class="fa-inline-code fa-code-green">VALID_REUSABLE</code> và <code class="fa-inline-code fa-code-green">RENDER_ONLY</code>, không chứa <code class="fa-inline-code fa-code-yellow">BLOCKED</code>. Vì nội dung đã current,
notebook không bị sửa chỉ để thay <code class="fa-inline-code">execution_count</code>.</p>
</section><section class="fa-section fa-debt-card"><h2 class="fa-heading fa-h2" id="reporting-debt-con-giu-nguyen"><span class="fa-section-pill">DEBT</span>Reporting debt còn giữ nguyên</h2>
<ul>
<li>Phase 23–30: saved processing logs mang historical label <code class="fa-inline-code fa-code-yellow">BLOCKED</code>, trong khi
current fail-closed inspection là <code class="fa-inline-code">VALID_REUSABLE/RENDER_ONLY</code>. Notebook dùng
live read-only audit nên không hiển thị sai.</li>
<li>Phase 37: thiếu <code class="fa-inline-code">s15_loss_sweep_manifest.json</code>, <code class="fa-inline-code">s15_loss_metrics.csv</code> và
canonical <code class="fa-inline-code">training_history.csv</code>.</li>
<li>Phase 38: thiếu <code class="fa-inline-code">s16_run_matrix.csv</code>, <code class="fa-inline-code">s16_epoch_cap_metrics.csv</code>,
<code class="fa-inline-code">s16_epoch_cap_effect.csv</code> và canonical <code class="fa-inline-code">training_history.csv</code>.</li>
<li>Phase 39: thiếu các CSV audit/reporting S17 và canonical
<code class="fa-inline-code">training_history.csv</code> được khai báo trong exact legacy contract.</li>
<li>Phase 40: thiếu <code class="fa-inline-code">s18_revin_metrics.csv</code> và canonical <code class="fa-inline-code">training_history.csv</code>.</li>
<li>Phase 41: thiếu <code class="fa-inline-code">s19_boundary_metrics.csv</code>, <code class="fa-inline-code">s19_boundary_winner.json</code> và
canonical <code class="fa-inline-code">training_history.csv</code>.</li>
<li>Phase 43: generic processing-log rebuild thiếu <code class="fa-inline-code">lstm_stage_lineage.csv</code>.</li>
<li>Phase 44: generic processing-log rebuild thiếu <code class="fa-inline-code">rolling_origin_results.csv</code>.</li>
<li>Phase 51: <code class="fa-inline-code">phase51_attention_handoff_cases.csv</code> và
<code class="fa-inline-code">phase51_tests_summary.csv</code> có expected SHA trong manifest nhưng chưa tìm thấy
source byte-verifiable; chúng không được fabricate.</li>
</ul>
<p>Các file trên là <code class="fa-inline-code fa-code-yellow">NON_BLOCKING</code> chỉ tại nơi exact legacy/current contract đã
chứng minh scientific core độc lập. Nếu core artifact hoặc SHA lệch, validator
vẫn fail-closed.</p>
</section><section class="fa-section fa-validation-card"><h2 class="fa-heading fa-h2" id="validation-cuoi"><span class="fa-section-pill">VALIDATION</span>Validation cuối</h2>
<div class="fa-table-wrap"><table class="fa-table">
<thead>
<tr>
<th>Validation</th>
<th>Result</th>
</tr>
</thead>
<tbody>
<tr>
<td>Focused reporting/recovery/Step16/Step17/selective-execution suite</td>
<td><code class="fa-inline-code">163 passed, 1 skipped</code></td>
</tr>
<tr>
<td>Phase 1–22 processing-log build</td>
<td><code class="fa-inline-code fa-code-green">PASS</code>/<code class="fa-inline-code fa-code-yellow">PASS_WITH_WARNING</code>, không exception</td>
</tr>
<tr>
<td>Phase 23–41 inspection</td>
<td><code class="fa-inline-code">19/19 VALID_REUSABLE</code>, action <code class="fa-inline-code fa-code-green">RENDER_ONLY</code></td>
</tr>
<tr>
<td>Phase 43–59 verified renderer</td>
<td><code class="fa-inline-code">17/17</code>, mỗi phase 3 bảng</td>
</tr>
<tr>
<td>Notebook</td>
<td>153 cells; 77/77 code cells có <code class="fa-inline-code">execution_count</code>; 0 saved Python error</td>
</tr>
<tr>
<td>Relative Markdown links</td>
<td>389 checked; 0 broken relative links</td>
</tr>
<tr>
<td>Scoped <code class="fa-inline-code">git diff --check</code> cho file sửa trong audit này</td>
<td><code class="fa-inline-code fa-code-green">PASS</code></td>
</tr>
<tr>
<td>Full <code class="fa-inline-code">git diff --check</code></td>
<td><code class="fa-inline-code">FAIL</code>: trailing whitespace/CRLF trong exact recovered historical JSON; không normalize vì sẽ đổi SHA</td>
</tr>
</tbody>
</table></div>
</section><section class="fa-section fa-conclusion-card"><h2 class="fa-heading fa-h2" id="ket-luan"><span class="fa-section-pill">CONCLUSION</span>Kết luận</h2><div class="fa-conclusion-strip"><div class="fa-conclusion-item"><b>STABLE_WITH_REPORTING_DEBT</b><span>Project scientific status</span></div><div class="fa-conclusion-item"><b>VALID_REUSABLE</b><span>Phase 34–41</span></div><div class="fa-conclusion-item"><b>NO TRAINING</b><span>Audit execution</span></div><div class="fa-conclusion-item"><b>NO TEST ACCESS</b><span>During audit</span></div></div>
<ul>
<li><code class="fa-inline-code fa-code-yellow">PROJECT_SCIENTIFIC_STATUS = STABLE_WITH_REPORTING_DEBT</code></li>
<li><code class="fa-inline-code fa-code-yellow">PHASE_1_59_STATUS = PASS_WITH_NON_BLOCKING_REPORTING_DEBT</code></li>
<li><code class="fa-inline-code">PHASE_34_41_STATUS = VALID_REUSABLE</code></li>
<li><code class="fa-inline-code">TRAINING_REQUIRED = NO</code></li>
<li><code class="fa-inline-code">TRAINING_EXECUTED = NO</code></li>
<li><code class="fa-inline-code">INFERENCE_EXECUTED = NO</code></li>
<li><code class="fa-inline-code">TEST_ACCESSED_DURING_AUDIT = NO</code></li>
<li><code class="fa-inline-code">SCIENTIFIC_EVIDENCE_OVERWRITTEN = NO</code></li>
</ul>
</section></div>
<div class="fa-footer"><h3>Audit integrity preserved</h3><p>Exact-checksum restoration · fail-closed verification · non-blocking reporting debt · no scientific evidence overwritten</p><div class="fa-floaters"><span class="fa-floater">🐍</span><span class="fa-floater">✓</span><span class="fa-floater">#️⃣</span><span class="fa-floater">📊</span><span class="fa-floater">🔒</span></div><svg class="fa-wave" viewBox="0 0 1200 120" preserveAspectRatio="none" aria-hidden="true"><path d="M0,70 C200,115 330,25 520,70 C710,115 860,28 1200,66 L1200,120 L0,120 Z" fill="#7dd3fc" opacity=".52"/><path d="M0,84 C220,35 390,122 600,78 C820,34 990,110 1200,75 L1200,120 L0,120 Z" fill="#f5c84b" opacity=".42"/></svg></div>
</div>
