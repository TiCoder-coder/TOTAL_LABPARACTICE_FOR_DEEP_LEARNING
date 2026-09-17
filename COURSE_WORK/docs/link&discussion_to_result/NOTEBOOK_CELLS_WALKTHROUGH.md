<style>
:root{--cw-ink:#111827;--cw-muted:#5f6b7a;--cw-blue:#0b66d6;--cw-blue-2:#38bdf8;--cw-blue-soft:#eef7ff;--cw-yellow:#f7c948;--cw-yellow-soft:#fff8d9;--cw-white:#fff;--cw-line:#dbe7f3;--cw-shadow:0 18px 50px rgba(17,24,39,.08);--cw-radius:24px}
*{box-sizing:border-box}.cw-app{max-width:1240px;margin:0 auto;padding:26px 28px 0;color:var(--cw-ink);font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;line-height:1.72;background:linear-gradient(180deg,#fff 0%,#fbfdff 55%,#fff 100%)}
.cw-app *{scroll-margin-top:24px}.cw-app a{transition:color .2s ease,background .2s ease,transform .2s ease,box-shadow .2s ease}.cw-link{color:var(--cw-blue);font-weight:650;text-decoration:none;border-bottom:1px solid rgba(11,102,214,.22)}.cw-link:hover{color:#084b9a;background:var(--cw-blue-soft);border-bottom-color:var(--cw-blue);border-radius:5px}
.cw-hero{position:relative;overflow:hidden;border:1px solid #d7e8f8;border-radius:34px;background:radial-gradient(circle at 9% 10%,rgba(56,189,248,.18),transparent 33%),radial-gradient(circle at 88% 10%,rgba(247,201,72,.22),transparent 31%),linear-gradient(135deg,#fff 0%,#f5faff 55%,#fffdf4 100%);box-shadow:0 30px 80px rgba(11,102,214,.12);margin:0 0 24px;isolation:isolate}.cw-hero-grid{position:absolute;inset:0;background-image:linear-gradient(rgba(11,102,214,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(11,102,214,.045) 1px,transparent 1px);background-size:28px 28px;mask-image:linear-gradient(to bottom,black,transparent 85%);z-index:-2}.cw-hero-orb{position:absolute;border-radius:50%;filter:blur(2px);opacity:.55;z-index:-1;animation:cw-drift 8s ease-in-out infinite}.cw-orb-a{width:170px;height:170px;right:-35px;top:-55px;background:rgba(247,201,72,.35)}.cw-orb-b{width:150px;height:150px;left:-50px;bottom:-70px;background:rgba(56,189,248,.26);animation-delay:-3s}.cw-hero-inner{padding:54px 54px 46px}.cw-eyebrow{display:inline-flex;align-items:center;gap:10px;padding:8px 13px;border-radius:999px;background:#fff;border:1px solid #dbeafe;color:#194f91;font-size:12px;font-weight:850;letter-spacing:.12em;box-shadow:0 8px 20px rgba(11,102,214,.07)}.cw-live-dot{width:8px;height:8px;border-radius:50%;background:var(--cw-yellow);box-shadow:0 0 0 6px rgba(247,201,72,.18);animation:cw-pulse 2s ease-in-out infinite}.cw-hero h1{max-width:940px;margin:20px 0 12px;font-size:clamp(34px,5.2vw,64px);line-height:1.04;letter-spacing:-.04em;color:#101828}.cw-hero-subtitle{max-width:900px;margin:0;color:#4b5b70;font-size:17px}.cw-tech-row{display:flex;flex-wrap:wrap;gap:10px;margin-top:26px}.cw-tech-chip{display:inline-flex;align-items:center;gap:9px;padding:9px 13px;border:1px solid #dbe8f5;border-radius:14px;background:rgba(255,255,255,.92);box-shadow:0 8px 20px rgba(17,24,39,.045);font-size:13px;cursor:default;transition:transform .22s ease,box-shadow .22s ease,border-color .22s ease}.cw-tech-chip:hover{transform:translateY(-3px);border-color:#a9d5ff;box-shadow:0 14px 28px rgba(11,102,214,.11)}.cw-svg-icon{display:inline-flex;width:28px;height:28px}.cw-svg-icon svg{width:100%;height:100%}.cw-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:30px}.cw-stat{padding:17px 18px;border-radius:18px;background:#fff;border:1px solid #dce9f5;box-shadow:0 8px 22px rgba(17,24,39,.04)}.cw-stat span{display:block;color:var(--cw-blue);font-weight:900;font-size:25px;line-height:1}.cw-stat small{display:block;margin-top:8px;color:#667085;font-weight:700}.cw-stat-policy{background:linear-gradient(135deg,#fff,#fff9dc);border-color:#f2db87}.cw-stat-policy span{color:#a56b00}
.cw-source-note{display:grid;grid-template-columns:auto 1fr;gap:16px;padding:20px 22px;margin:24px 0 30px;border:1px solid #cfe6fb;border-left:5px solid var(--cw-blue);border-radius:20px;background:linear-gradient(90deg,#f3f9ff,#fff);box-shadow:0 12px 28px rgba(17,24,39,.045)}.cw-source-note-icon{display:grid;place-items:center;width:34px;height:34px;border-radius:10px;background:var(--cw-blue);color:#fff;font-weight:900}.cw-source-note-title{font-weight:900;color:#123f73;margin-bottom:4px}.cw-source-note blockquote{margin:0;border:0;padding:0;background:none}.cw-source-note p{margin:5px 0;color:#4b5b70}
.cw-content{display:block}.cw-section{position:relative;margin:30px 0;padding:28px;border:1px solid var(--cw-line);border-radius:var(--cw-radius);background:rgba(255,255,255,.97);box-shadow:var(--cw-shadow);transition:transform .26s ease,box-shadow .26s ease,border-color .26s ease}.cw-section:hover{transform:translateY(-2px);box-shadow:0 24px 60px rgba(11,102,214,.10);border-color:#c3dcf3}.cw-section:before{content:"";position:absolute;left:0;top:28px;bottom:28px;width:4px;border-radius:10px;background:linear-gradient(180deg,var(--cw-blue),var(--cw-blue-2),var(--cw-yellow))}.cw-section-heading{display:flex;align-items:flex-start;gap:14px;margin-bottom:18px}.cw-section-index{flex:0 0 auto;display:grid;place-items:center;min-width:44px;height:44px;padding:0 10px;border-radius:14px;background:linear-gradient(135deg,var(--cw-blue),#2094ef);color:#fff;font-weight:900;box-shadow:0 10px 24px rgba(11,102,214,.20)}.cw-section h2{margin:2px 0 0;font-size:clamp(24px,3.1vw,36px);line-height:1.18;letter-spacing:-.025em;color:#132238}.cw-section-final,.cw-section-v2{background:linear-gradient(180deg,#fff,#fbfdff 72%,#fffdf2)}.cw-section-final:after,.cw-section-v2:after{content:"FINAL EVIDENCE";position:absolute;right:22px;top:22px;padding:5px 10px;border-radius:999px;background:var(--cw-yellow-soft);color:#9b6500;font-size:10px;font-weight:900;letter-spacing:.08em;border:1px solid #f1d36e}.cw-section-lineage{background:linear-gradient(135deg,#f9fcff,#fff)}
.cw-card{position:relative;margin:18px 0 0;padding:20px 22px;border:1px solid #e1eaf3;border-radius:19px;background:#fff;box-shadow:0 8px 24px rgba(17,24,39,.035);overflow:hidden;transition:border-color .2s ease,box-shadow .2s ease,transform .2s ease}.cw-card:hover{transform:translateX(3px);border-color:#bcd9f5;box-shadow:0 12px 30px rgba(11,102,214,.07)}.cw-cell-card{padding-left:28px}.cw-card-rail{position:absolute;left:0;top:0;bottom:0;width:7px;background:linear-gradient(180deg,var(--cw-blue),var(--cw-blue-2));opacity:.92}.cw-card-dot{position:absolute;left:1px;top:27px;width:5px;height:5px;border-radius:50%;background:var(--cw-yellow);box-shadow:0 0 0 5px rgba(247,201,72,.22)}.cw-card h3{margin:0 0 13px;font-size:20px;line-height:1.35;color:#17263a}.cw-card-kicker{display:block;margin-bottom:7px;color:#2f78bf;font-size:10px;font-weight:900;letter-spacing:.12em}.cw-subcard{background:linear-gradient(180deg,#fff,#fbfdff)}
.cw-p{margin:10px 0;color:#29384a}.cw-strong{color:#17263a}.cw-list{margin:10px 0 12px;padding-left:22px}.cw-li{margin:6px 0;color:#2c3a4b}.cw-li::marker{color:var(--cw-blue)}.cw-section-toc .cw-card{background:#fbfdff}.cw-section-toc .cw-list{columns:1}.cw-callout{margin:16px 0;padding:15px 17px;border:1px solid #f1d77a;border-left:5px solid var(--cw-yellow);border-radius:14px;background:linear-gradient(90deg,var(--cw-yellow-soft),#fff);color:#554313}.cw-rule{height:1px;border:0;background:linear-gradient(90deg,transparent,#d6e5f3 12%,#d6e5f3 88%,transparent);margin:24px 0}.cw-code{padding:.16em .42em;border-radius:7px;background:#f1f6fb;color:#134c82;font-size:.92em;border:1px solid #e0ebf5}.cw-code-block{overflow:auto;padding:17px;border-radius:15px;background:#0f1a2a;color:#e7f1fb;box-shadow:inset 0 0 0 1px rgba(255,255,255,.07)}.cw-code-block .cw-code{padding:0;border:0;background:transparent;color:inherit}
.cw-table-wrap{overflow-x:auto;margin:16px 0;border:1px solid #dce8f3;border-radius:16px;background:#fff;box-shadow:0 8px 25px rgba(17,24,39,.04)}.cw-table{width:100%;border-collapse:separate;border-spacing:0;font-size:14px;min-width:680px}.cw-table th{padding:12px 14px;text-align:left;background:linear-gradient(180deg,#edf7ff,#e8f4ff);color:#173c63;border-bottom:1px solid #cfe3f4;font-weight:850;white-space:nowrap}.cw-table td{padding:11px 14px;border-bottom:1px solid #edf2f7;color:#2c3b4d;vertical-align:top}.cw-table tr:last-child td{border-bottom:0}.cw-table tbody tr:nth-child(even) td{background:#fcfdff}.cw-table tbody tr:hover td{background:#fffbea}.cw-semantic-note{padding:10px 12px;border-radius:11px;background:#f7fbff;border:1px solid #e1edf7}
.cw-footer{position:relative;overflow:hidden;margin:48px -28px 0;min-height:410px;padding:54px 42px 190px;background:linear-gradient(180deg,#f8fcff 0%,#eef8ff 50%,#fff6c7 100%);border-top:1px solid #d9ebf8}.cw-footer-content{position:relative;z-index:4;max-width:780px;margin:0 auto;text-align:center}.cw-footer-kicker{color:#1d6fc1;font-weight:900;letter-spacing:.12em;font-size:11px}.cw-footer h2{margin:10px 0 10px;font-size:clamp(25px,4vw,42px);line-height:1.16;color:#12243b;letter-spacing:-.03em}.cw-footer p{color:#57677a}.cw-backtop{display:inline-flex;margin-top:14px;padding:10px 15px;border-radius:999px;background:#fff;color:var(--cw-blue);border:1px solid #cfe6fb;text-decoration:none;font-weight:850;box-shadow:0 8px 20px rgba(11,102,214,.08)}.cw-backtop:hover{transform:translateY(-2px);box-shadow:0 12px 28px rgba(11,102,214,.13)}.cw-floating-icons{position:absolute;inset:auto 0 88px;z-index:5;display:flex;justify-content:center;gap:42px;pointer-events:none}.cw-floating-icons span{display:grid;place-items:center;width:45px;height:45px;border-radius:15px;background:rgba(255,255,255,.9);border:1px solid rgba(11,102,214,.18);box-shadow:0 10px 26px rgba(17,24,39,.09);font-size:22px;animation:cw-float 4.2s ease-in-out infinite;animation-delay:calc(var(--i) * -.65s)}.cw-wave{position:absolute;left:-2%;bottom:-1px;width:104%;height:190px;z-index:2}.cw-wave-blue{fill:#2c90e8;opacity:.70;animation:cw-wave 7s ease-in-out infinite}.cw-wave-yellow{fill:#f7c948;opacity:.92;animation:cw-wave 8.5s ease-in-out infinite reverse}.cw-wave-white{fill:#fff;opacity:.96}
@keyframes cw-pulse{0%,100%{transform:scale(1);box-shadow:0 0 0 6px rgba(247,201,72,.17)}50%{transform:scale(1.25);box-shadow:0 0 0 10px rgba(247,201,72,.08)}}@keyframes cw-drift{0%,100%{transform:translate3d(0,0,0)}50%{transform:translate3d(-14px,12px,0)}}@keyframes cw-float{0%,100%{transform:translateY(0) rotate(0)}50%{transform:translateY(-13px) rotate(3deg)}}@keyframes cw-wave{0%,100%{transform:translateX(0) scaleY(1)}50%{transform:translateX(-1.3%) scaleY(1.06)}}
@media (max-width:900px){.cw-app{padding:18px 14px 0}.cw-hero-inner{padding:38px 25px 32px}.cw-stats{grid-template-columns:repeat(2,1fr)}.cw-section{padding:22px 18px}.cw-footer{margin-left:-14px;margin-right:-14px}.cw-section-final:after,.cw-section-v2:after{position:static;display:inline-block;margin:0 0 10px 58px}.cw-tech-row{gap:8px}.cw-tech-chip{padding:8px 10px}.cw-floating-icons{gap:18px}}@media (max-width:560px){.cw-stats{grid-template-columns:1fr 1fr}.cw-hero h1{font-size:34px}.cw-section-heading{gap:10px}.cw-section-index{min-width:38px;height:38px}.cw-card{padding:17px}.cw-cell-card{padding-left:23px}.cw-source-note{grid-template-columns:1fr}.cw-floating-icons span{width:38px;height:38px;font-size:18px}.cw-footer{padding-left:20px;padding-right:20px}.cw-table{font-size:13px}}@media (prefers-reduced-motion:reduce){.cw-app *{animation:none!important;transition:none!important;scroll-behavior:auto!important}}
</style><div class="cw-app">
<header class="cw-hero" id="cw-top">
<div class="cw-hero-grid"></div>
<div class="cw-hero-orb cw-orb-a"></div>
<div class="cw-hero-orb cw-orb-b"></div>
<div class="cw-hero-inner">
<div class="cw-eyebrow"><span class="cw-live-dot"></span> COURSEWORK · NOTEBOOK DOCUMENTATION</div>
<h1>CourseWork Notebook — Cell Walkthrough &amp; Output Guide</h1>
<p class="cw-hero-subtitle">Interactive visual walkthrough cho toàn bộ <strong>153 cells</strong>, từ Foundation đến MODEL_IMPROVEMENT_V2, với lineage và artifact contract được giữ nguyên.</p>
<div aria-label="Technology stack" class="cw-tech-row">
<span class="cw-tech-chip"><span class="cw-svg-icon"><svg aria-hidden="true" viewbox="0 0 64 64"><path d="M31.8 7.5c-12.7 0-11.9 5.5-11.9 5.5v5.7H32v1.8H15.1S7 19.6 7 32.3 14 44.5 14 44.5h4.2v-5.9s-.2-7 6.9-7h12c6.6 0 6.6-6.4 6.6-6.4V13.5S44.7 7.5 31.8 7.5Zm-6.7 4.1a2.3 2.3 0 1 1 0 4.6 2.3 2.3 0 0 1 0-4.6Z" fill="#3776AB"></path><path d="M32.2 56.5c12.7 0 11.9-5.5 11.9-5.5v-5.7H32v-1.8h16.9S57 44.4 57 31.7 50 19.5 50 19.5h-4.2v5.9s.2 7-6.9 7h-12c-6.6 0-6.6 6.4-6.6 6.4v11.7s-1 6 11.9 6Zm6.7-4.1a2.3 2.3 0 1 1 0-4.6 2.3 2.3 0 0 1 0 4.6Z" fill="#FFD43B"></path></svg></span><b>Python</b></span>
<span class="cw-tech-chip"><span class="cw-svg-icon"><svg aria-hidden="true" viewbox="0 0 64 64"><path d="M35 6c3 12-8 15-3 25 3-5 8-7 9-14 8 8 12 17 8 27-3 8-10 14-19 14-12 0-21-9-21-21 0-11 7-19 15-26-1 8 1 13 5 15 0-9 2-15 6-20Z" fill="#0b66d6"></path><path d="M33 31c5 5 8 10 5 16-2 4-6 6-10 5-6-1-9-8-6-13 2-4 5-6 8-9 0 4 1 6 3 8 1-2 1-4 0-7Z" fill="#f7c948"></path></svg></span><b>PyTorch</b></span>
<span class="cw-tech-chip"><span class="cw-svg-icon"><svg aria-hidden="true" viewbox="0 0 64 64"><rect fill="#fff" height="46" rx="7" stroke="#0b66d6" stroke-width="4" width="40" x="12" y="9"></rect><path d="M20 20h24M20 30h24M20 40h16" stroke="#0b66d6" stroke-linecap="round" stroke-width="4"></path><path d="M9 18h8M9 30h8M9 42h8" stroke="#f7c948" stroke-linecap="round" stroke-width="4"></path></svg></span><b>Jupyter</b></span>
<span class="cw-tech-chip"><span class="cw-svg-icon"><svg aria-hidden="true" viewbox="0 0 64 64"><path d="M32 8 45 21 32 34 19 21 32 8Zm0 22 13 13-13 13-13-13 13-13Z" fill="none" stroke="#0b66d6" stroke-width="4"></path><circle cx="32" cy="21" fill="#f7c948" r="4"></circle><circle cx="32" cy="43" fill="#f7c948" r="4"></circle></svg></span><b>Transformer</b></span>
<span class="cw-tech-chip"><span class="cw-svg-icon"><svg aria-hidden="true" viewbox="0 0 64 64"><path d="M10 50h44" stroke="#111827" stroke-linecap="round" stroke-width="4"></path><path d="m12 43 11-12 10 6 10-17 10 7" fill="none" stroke="#0b66d6" stroke-linecap="round" stroke-linejoin="round" stroke-width="4"></path><circle cx="43" cy="20" fill="#f7c948" r="4"></circle></svg></span><b>Time Series</b></span>
<span class="cw-tech-chip"><span class="cw-svg-icon"><svg aria-hidden="true" viewbox="0 0 64 64"><circle cx="32" cy="32" fill="#eef8ff" r="23" stroke="#0b66d6" stroke-width="4"></circle><path d="m21 32 8 8 15-18" fill="none" stroke="#0b66d6" stroke-linecap="round" stroke-linejoin="round" stroke-width="5"></path><circle cx="48" cy="16" fill="#f7c948" r="6"></circle></svg></span><b>Verified Artifacts</b></span>
</div>
<div class="cw-stats">
<div class="cw-stat"><span>153</span><small>Notebook Cells</small></div>
<div class="cw-stat"><span>59</span><small>Phases</small></div>
<div class="cw-stat"><span>3</span><small>Final Seeds</small></div>
<div class="cw-stat cw-stat-policy"><span>V2</span><small>Final Locked Policy</small></div>
</div>
</div>
</header>

<div class="cw-content"><section class="cw-section cw-section-toc" data-section="1"><div class="cw-section-heading"><span class="cw-section-index">01</span><h2 class="cw-heading cw-h2">Mục Lục</h2></div><article class="cw-card cw-subcard"><h3 class="cw-heading cw-h3">Phần Khai Báo</h3><ul class="cw-list">
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 0: Title</a> — ID <code class="cw-code">coursework-title</code> · <a class="cw-link" href="#cell-0">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 1: Define Problem</a> — ID <code class="cw-code">define-problem</code> · <a class="cw-link" href="#cell-1">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 2: Imports</a> — ID <code class="cw-code">0caba13e</code> · <a class="cw-link" href="#cell-2">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 3: Display setup</a> — ID <code class="cw-code">public-api-imports</code> · <a class="cw-link" href="#cell-3">Giải thích</a></li>
</ul></article><article class="cw-card cw-subcard"><h3 class="cw-heading cw-h3">Giai Đoạn Foundation (Phases 1–11)</h3><ul class="cw-list">
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 4–6: Phase 1 - Environment</a> — output Cell 5, ID <code class="cw-code">phase-1-orchestration</code> · <a class="cw-link" href="#cell-4">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 7–8: Phase 2 - Data Acquisition</a> — output Cell 8, ID <code class="cw-code">phase-2-orchestration</code> · <a class="cw-link" href="#cell-7">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 9–10: Phase 3 - Schema Audit</a> — output Cell 10, ID <code class="cw-code">phase-3-orchestration</code> · <a class="cw-link" href="#cell-9">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 11–12: Phase 4 - Temporal Integrity Audit</a> — output Cell 12, ID <code class="cw-code">phase-4-orchestration</code> · <a class="cw-link" href="#cell-11">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 13–14: Phase 5 - Chronological Split</a> — output Cell 14, ID <code class="cw-code">phase-5-orchestration</code> · <a class="cw-link" href="#cell-13">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 15–44: Phase 6 - Exploratory Data Analysis</a> — output Cell 16, ID <code class="cw-code">phase-6-setup</code> · <a class="cw-link" href="#cell-15">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 45–46: Phase 7 - Feature Engineering</a> — output Cell 46, ID <code class="cw-code">phase-7-orchestration</code> · <a class="cw-link" href="#cell-45">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 47–48: Phase 8 - Feature-Set Variants</a> — output Cell 48, ID <code class="cw-code">phase-8-orchestration</code> · <a class="cw-link" href="#cell-47">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 49–50: Phase 9 - Train-Only Scaling</a> — output Cell 50, ID <code class="cw-code">phase-9-orchestration</code> · <a class="cw-link" href="#cell-49">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 51–52: Phase 10 - Window Builder</a> — output Cell 52, ID <code class="cw-code">phase-10-orchestration</code> · <a class="cw-link" href="#cell-51">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 53–54: Phase 11 - DataLoaders</a> — output Cell 54, ID <code class="cw-code">phase-11-orchestration</code> · <a class="cw-link" href="#cell-53">Giải thích</a></li>
</ul></article><article class="cw-card cw-subcard"><h3 class="cw-heading cw-h3">Giai Đoạn Modeling (Phases 12–22)</h3><ul class="cw-list">
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 55–56: Phase 12 - Shared Metrics</a> — output Cell 56, ID <code class="cw-code">phase-12-orchestration</code> · <a class="cw-link" href="#cell-55">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 57–58: Phase 13 - Experiment Registry</a> — output Cell 58, ID <code class="cw-code">phase-13-orchestration</code> · <a class="cw-link" href="#cell-57">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 59–60: Phase 14 - Persistence Baseline</a> — output Cell 60, ID <code class="cw-code">phase-14-orchestration</code> · <a class="cw-link" href="#cell-59">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 61–62: Phase 15 - LSTM Implementation</a> — output Cell 62, ID <code class="cw-code">499bc911</code> · <a class="cw-link" href="#cell-61">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 63–64: Phase 16 - Transformer Implementation</a> — output Cell 64, ID <code class="cw-code">518fca61</code> · <a class="cw-link" href="#cell-63">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 65–66: Phase 17 - Attention-Aware Encoder Verification</a> — output Cell 66, ID <code class="cw-code">efc7f487</code> · <a class="cw-link" href="#cell-65">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 67–68: Phase 18 - Forward-Pass Sanity Tests</a> — output Cell 68, ID <code class="cw-code">d60e1951</code> · <a class="cw-link" href="#cell-67">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 69–70: Phase 19 - Baseline Training Engine</a> — output Cell 70, ID <code class="cw-code">2acc38f9</code> · <a class="cw-link" href="#cell-69">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 71–72: Phase 20 - LSTM Baseline Run</a> — output Cell 72, ID <code class="cw-code">2eb13c9c</code> · <a class="cw-link" href="#cell-71">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 73–74: Phase 21 - Transformer B0 Run</a> — output Cell 74, ID <code class="cw-code">03db9e36</code> · <a class="cw-link" href="#cell-73">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 75–76: Phase 22 - Learning-Curve Diagnostics</a> — output Cell 76, ID <code class="cw-code">cd4716af</code> · <a class="cw-link" href="#cell-75">Giải thích</a></li>
</ul></article><article class="cw-card cw-subcard"><h3 class="cw-heading cw-h3">Giai Đoạn Sweeps (Phases 23–41)</h3><ul class="cw-list">
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 77–78: Phase 23 - S1 Feature-Set Sweep</a> — output Cell 78, ID <code class="cw-code">0881cfe3</code> · <a class="cw-link" href="#cell-77">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 79–80: Phase 24 - S2 Time-Feature Sweep</a> — output Cell 80, ID <code class="cw-code">9b2d6e88</code> · <a class="cw-link" href="#cell-79">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 81–82: Phase 25 - S3 Target-Scaling Sweep</a> — output Cell 82, ID <code class="cw-code">51bc5965</code> · <a class="cw-link" href="#cell-81">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 83–84: Phase 26 - S4 Lookback Sweep</a> — output Cell 84, ID <code class="cw-code">585b390d</code> · <a class="cw-link" href="#cell-83">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 85–86: Phase 27 - S5 Pooling Sweep</a> — output Cell 86, ID <code class="cw-code">86f4ac0c</code> · <a class="cw-link" href="#cell-85">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 87–88: Phase 28 - S6 Activation Sweep</a> — output Cell 88, ID <code class="cw-code">b70c9707</code> · <a class="cw-link" href="#cell-87">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 89–90: Phase 29 - S7 Batch-Size Sweep</a> — output Cell 90, ID <code class="cw-code">eb4f1b80</code> · <a class="cw-link" href="#cell-89">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 91–92: Phase 30 - S8 Learning-Rate Sweep</a> — output Cell 92, ID <code class="cw-code">3fe4478c</code> · <a class="cw-link" href="#cell-91">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 93–94: Phase 31 - S9 Weight-Decay Sweep</a> — output Cell 94, ID <code class="cw-code">phase-31-resume</code> · <a class="cw-link" href="#cell-93">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 95–96: Phase 32 - S10 Dropout Sweep</a> — output Cell 96, ID <code class="cw-code">phase-32-resume</code> · <a class="cw-link" href="#cell-95">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 97–98: Transformer Configuration after Phase 33</a> — output Cell 98, ID <code class="cw-code">phase-33-config-display</code> · <a class="cw-link" href="#cell-97">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 99–100: Phase 34 - S12 Head Sweep</a> — output Cell 100, ID <code class="cw-code">phase-34-resume</code> · <a class="cw-link" href="#cell-99">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 101–102: Phase 35 - S13 Layer Sweep</a> — output Cell 102, ID <code class="cw-code">phase-35-resume</code> · <a class="cw-link" href="#cell-101">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 103–104: Phase 36 - S14 FFN Sweep</a> — output Cell 104, ID <code class="cw-code">phase-36-resume</code> · <a class="cw-link" href="#cell-103">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 105–106: Phase 37 - S15 Loss Sweep</a> — output Cell 106, ID <code class="cw-code">phase-37-resume</code> · <a class="cw-link" href="#cell-105">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 107–108: Phase 38 - S16 Epoch-Cap Sweep</a> — output Cell 108, ID <code class="cw-code">17dc5a66</code> · <a class="cw-link" href="#cell-107">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 109–110: Phase 39 - S17 Gradient-Clipping Sweep</a> — output Cell 110, ID <code class="cw-code">f0fcdce7</code> · <a class="cw-link" href="#cell-109">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 111–112: Phase 40 - S18 RevIN Sweep</a> — output Cell 112, ID <code class="cw-code">d407f95c</code> · <a class="cw-link" href="#cell-111">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 113–114: Phase 41 - S19 Boundary-Protocol Check</a> — output Cell 114, ID <code class="cw-code">7e86663d</code> · <a class="cw-link" href="#cell-113">Giải thích</a></li>
</ul></article><article class="cw-card cw-subcard"><h3 class="cw-heading cw-h3">Giai Đoạn Final Pipeline và Benchmark Presentation (Phases 42–47)</h3><ul class="cw-list">
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 115–116: Phase 42 - Candidate Synthesis</a> — output Cell 116, ID <code class="cw-code">994bbdb9</code> · <a class="cw-link" href="#cell-115">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 117–118: Phase 43 — LSTM Tuning Results</a> — output Cell 118, ID <code class="cw-code">6a082d01</code> · <a class="cw-link" href="#cell-117">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 119–120: Phase 44 — Rolling-Origin Robustness Results</a> — output Cell 120, ID <code class="cw-code">2f0130ea</code> · <a class="cw-link" href="#cell-119">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 121–122: Phase 45 — Final Model Lock Results</a> — output Cell 122, ID <code class="cw-code">4e6b273d</code> · <a class="cw-link" href="#cell-121">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 123–124: Phase 46 — Three-Seed Final Run Results</a> — output Cell 124, ID <code class="cw-code">753ab0f1</code> · <a class="cw-link" href="#cell-123">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 125–126: Phase 47 — Final V2 Post-hoc Benchmark Results</a> — output Cell 126, ID <code class="cw-code">4dea0c14</code> · <a class="cw-link" href="#cell-125">Giải thích</a></li>
</ul></article><article class="cw-card cw-subcard"><h3 class="cw-heading cw-h3">Giai Đoạn Analysis (Phases 48–59)</h3><ul class="cw-list">
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 127–128: Phase 48 — Final V2 Prediction Analysis Results</a> — output Cell 128, ID <code class="cw-code">1451ccf0</code> · <a class="cw-link" href="#cell-127">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 129–130: Phase 49 — Final V2 Residual Analysis Results</a> — output Cell 130, ID <code class="cw-code">bbcf08f2</code> · <a class="cw-link" href="#cell-129">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 131–132: Phase 50 — Final V2 Error-by-Regime Analysis Results</a> — output Cell 132, ID <code class="cw-code">1d5b5327</code> · <a class="cw-link" href="#cell-131">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 133–134: Phase 51 — Final V2 Worst-Error Analysis Results</a> — output Cell 134, ID <code class="cw-code">c9df346a</code> · <a class="cw-link" href="#cell-133">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 135–136: Phase 52 — Historical V1 Attention Extraction Results</a> — output Cell 136, ID <code class="cw-code">7b48186f</code> · <a class="cw-link" href="#cell-135">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 137–138: Phase 53 — Historical V1 Attention Heatmap Results</a> — output Cell 138, ID <code class="cw-code">1c098fa6</code> · <a class="cw-link" href="#cell-137">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 139–140: Phase 54 — Historical V1 Last-Query Attention Results</a> — output Cell 140, ID <code class="cw-code">ec7b2ea5</code> · <a class="cw-link" href="#cell-139">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 141–142: Phase 55 — Historical V1 Head Comparison Results</a> — output Cell 142, ID <code class="cw-code">05ec07a2</code> · <a class="cw-link" href="#cell-141">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 143–144: Phase 56 — Historical V1 Error-Conditioned Attention Results</a> — output Cell 144, ID <code class="cw-code">5e9eec01</code> · <a class="cw-link" href="#cell-143">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 145–146: Phase 57 — Historical V1 Seed-Stability Attention Results</a> — output Cell 146, ID <code class="cw-code">4293c1aa</code> · <a class="cw-link" href="#cell-145">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 147–148: Phase 58 — Final V2 Results Summary</a> — output Cell 148, ID <code class="cw-code">9f864b91</code> · <a class="cw-link" href="#cell-147">Giải thích</a></li>
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 149–150: Phase 59 — Final V2 Conclusions</a> — output Cell 150, ID <code class="cw-code">76de7c69</code> · <a class="cw-link" href="#cell-149">Giải thích</a></li>
</ul></article><article class="cw-card cw-subcard"><h3 class="cw-heading cw-h3">MODEL_IMPROVEMENT_V2</h3><ul class="cw-list">
<li class="cw-li"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 151–152: MODEL_IMPROVEMENT_V2 — Results Summary</a> — IDs <code class="cw-code">v2closure</code> → <code class="cw-code">v2closedash</code> · <a class="cw-link" href="#model-improvement-v2">Giải thích</a></li>
</ul><hr class="cw-rule"/></article></section><section class="cw-section cw-section-lineage" data-section="2"><div class="cw-section-heading"><span class="cw-section-index">02</span><h2 class="cw-heading cw-h2">Lineage hiện tại</h2></div><ul class="cw-list">
<li class="cw-li">Phase 1–46: V1; Phase 34–41 là verified recovered historical lineage.</li>
<li class="cw-li">Phase 47–51: <code class="cw-code">V2 FINAL</code>.</li>
<li class="cw-li">Phase 52–57: <code class="cw-code">V1 HISTORICAL ATTENTION</code>; không phải evidence attention của final V2.</li>
<li class="cw-li">Phase 58–59: <code class="cw-code">V2 FINAL</code>.</li>
<li class="cw-li">Historical V1 artifacts vẫn được giữ cho provenance; presentation không overwrite scientific evidence.</li>
</ul><hr class="cw-rule"/></section><section class="cw-section" data-section="3"><div class="cw-section-heading"><span class="cw-section-index">03</span><h2 class="cw-heading cw-h2">Phần Khai Báo (Cells 0-3)</h2></div><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">NOTEBOOK WALKTHROUGH</span><a id="cell-0"></a>Cell 0 — Title</h3><p class="cw-p"><strong class="cw-strong">Cell ID:</strong> <code class="cw-code">coursework-title</code>
<strong class="cw-strong">Loại:</strong> Markdown</p><p class="cw-p"><strong class="cw-strong">Nội dung:</strong> Tiêu đề <code class="cw-code"># MULTIVARIATE TIME-SERIES REGRESSION</code> cho notebook.</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Đây là markdown cell đầu tiên, không có output.</li>
<li class="cw-li">Dùng để giới thiệu tổng quan project là <strong class="cw-strong">Multivariate Time-Series Regression</strong> trên dataset UCI Appliances Energy Prediction.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 0 trong notebook</a> → tìm cell có text <code class="cw-code"># MULTIVARIATE TIME-SERIES REGRESSION</code></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">NOTEBOOK WALKTHROUGH</span><a id="cell-1"></a>Cell 1 — Define Problem</h3><p class="cw-p"><strong class="cw-strong">Cell ID:</strong> <code class="cw-code">define-problem</code>
<strong class="cw-strong">Loại:</strong> Markdown</p><p class="cw-p"><strong class="cw-strong">Nội dung:</strong> Markdown <code class="cw-code">## Define Problem</code>.</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Markdown cell định nghĩa bài toán, không có output.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 1 trong notebook</a> → tìm cell có text <code class="cw-code">## Define Problem</code></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">NOTEBOOK WALKTHROUGH</span><a id="cell-2"></a>Cell 2 — Imports</h3><p class="cw-p"><strong class="cw-strong">Cell ID:</strong> <code class="cw-code">0caba13e</code>
<strong class="cw-strong">Loại:</strong> Code</p><p class="cw-p"><strong class="cw-strong">Nội dung:</strong> <code class="cw-code">import sys</code>, các imports hệ thống.</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng:</strong> Không có output stdout (chỉ là imports). Nếu có lỗi sẽ xuất <code class="cw-code">ModuleNotFoundError</code> hoặc <code class="cw-code">ImportError</code>.</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Import <code class="cw-code">sys</code> và các thiết lập path.</li>
<li class="cw-li">Đây là cell import đầu tiên, nếu fail sẽ chặn toàn bộ pipeline.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 2 trong notebook</a> → bấm vào cell chứa <code class="cw-code">import sys</code></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">NOTEBOOK WALKTHROUGH</span><a id="cell-3"></a>Cell 3 — Display Setup</h3><p class="cw-p"><strong class="cw-strong">Cell ID:</strong> <code class="cw-code">public-api-imports</code>
<strong class="cw-strong">Loại:</strong> Code</p><p class="cw-p"><strong class="cw-strong">Nội dung:</strong> <code class="cw-code">from IPython.display import Image, display</code>.</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng:</strong> Không có output. Chỉ là import.</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Import <code class="cw-code">Image</code> và <code class="cw-code">display</code> từ IPython để hiển thị ảnh PNG (EDA figures, heatmaps) inline trong notebook.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 3 trong notebook</a> → bấm vào cell chứa <code class="cw-code">from IPython.display import Image, display</code></p><hr class="cw-rule"/></article></section><section class="cw-section" data-section="4"><div class="cw-section-heading"><span class="cw-section-index">04</span><h2 class="cw-heading cw-h2">Giai Đoạn Foundation (Cells 4–54)</h2></div><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 1 · NOTEBOOK WALKTHROUGH</span><a id="cell-4"></a>Cell 4–6 — Phase 1 - Environment</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 4–6</a> — heading ID <code class="cw-code">phase-1-heading</code>, output cell 5 ID <code class="cw-code">phase-1-orchestration</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="cw-code">render_phase_summary(1, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/environment/phase_1_signoff.json">phase_1_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.</p><p class="cw-p"><strong class="cw-strong">Cell 6 (code) ID:</strong> <code class="cw-code">b0f56164</code> — environment audit only; dependency installation bị loại khỏi <code class="cw-code">Run All</code>.
<strong class="cw-strong">Cell 4 (markdown):</strong> <code class="cw-code">phase-1-heading</code> ID, tiêu đề <code class="cw-code">## Phase 1 - Environment</code>
<strong class="cw-strong">Cell 5 (code):</strong> <code class="cw-code">phase-1-orchestration</code> ID, chạy <code class="cw-code">render_phase_summary(1, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng của Cell 5:</strong></p><ul class="cw-list">
<li class="cw-li">In ra một dictionary với environment fingerprint (Python version, torch version, deterministic mode, ...).</li>
<li class="cw-li">Ghi file <code class="cw-code">artifacts/environment/environment_report.json</code>.</li>
<li class="cw-li">Ghi file <code class="cw-code">artifacts/environment/phase_1_signoff.json</code> với status <code class="cw-code">PASS</code>.</li>
</ul><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Phase 1 capture environment, freeze requirements, smoke test imports.</li>
<li class="cw-li">Nếu output báo lỗi về kernel hoặc import, kiểm tra lại <code class="cw-code">requirements_freeze.txt</code>.</li>
<li class="cw-li">Sign-off <code class="cw-code">PASS</code> là điều kiện tiên quyết để chạy Phase 2.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 5 trong notebook</a> → tìm cell ID <code class="cw-code">phase-1-orchestration</code>.</p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 2 · NOTEBOOK WALKTHROUGH</span><a id="cell-7"></a>Cell 7–8 — Phase 2 - Data Acquisition</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 7–8</a> — heading ID <code class="cw-code">phase-2-heading</code>, output cell 8 ID <code class="cw-code">phase-2-orchestration</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="cw-code">render_frozen_phase_evidence(2, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/acquisition/phase_2_signoff.json">phase_2_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 7 (markdown):</strong> Tiêu đề <code class="cw-code">## Phase 2 - Data Acquisition</code>
<strong class="cw-strong">Cell 8 (code):</strong> Chạy <code class="cw-code">render_frozen_phase_evidence(2, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng của Cell 8:</strong></p><ul class="cw-list">
<li class="cw-li">Dictionary chứa dataset manifest: rows, columns, source SHA256, acquisition timestamp.</li>
<li class="cw-li">File <code class="cw-code">data/raw_data/energydata_complete.csv</code> được materialize.</li>
<li class="cw-li">File <code class="cw-code">artifacts/acquisition/dataset_manifest.json</code> và <code class="cw-code">phase_2_signoff.json</code> được tạo.</li>
</ul><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Phase kiểm tra SHA256 checksum của raw data, đảm bảo dữ liệu đúng với nguồn.</li>
<li class="cw-li">Nếu checksum mismatch, xem file <code class="cw-code">phase_2_signoff.json</code> để biết lý do.</li>
<li class="cw-li">Output thường có <code class="cw-code">n_rows ≈ 19735</code> (mẫu 10 phút × ~4.5 tháng).</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 8 trong notebook</a> → tìm cell ID <code class="cw-code">phase-2-orchestration</code>.</p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 3 · NOTEBOOK WALKTHROUGH</span><a id="cell-9"></a>Cell 9–10 — Phase 3 - Schema Audit</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 9–10</a> — heading ID <code class="cw-code">phase-3-heading</code>, output cell 10 ID <code class="cw-code">phase-3-orchestration</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="cw-code">render_phase_summary(3, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/schema/phase_3_signoff.json">phase_3_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 9 (markdown):</strong> Tiêu đề <code class="cw-code">## Phase 3 - Schema Audit</code>
<strong class="cw-strong">Cell 10 (code):</strong> Chạy <code class="cw-code">render_phase_summary(3, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng của Cell 10:</strong></p><ul class="cw-list">
<li class="cw-li">Manifest: 29 columns, dtypes cho mỗi biến, missing counts.</li>
<li class="cw-li">File <code class="cw-code">artifacts/schema/{schema_manifest.json, schema_summary.csv, variable_dictionary.csv}</code>.</li>
</ul><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Schema audit đảm bảo dtypes và cấu trúc cột khớp với contract.</li>
<li class="cw-li"><code class="cw-code">variable_dictionary.csv</code> chứa metadata cho mỗi biến (đơn vị, role, range).</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 10 trong notebook</a> → tìm cell ID <code class="cw-code">phase-3-orchestration</code>.</p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 4 · NOTEBOOK WALKTHROUGH</span><a id="cell-11"></a>Cell 11–12 — Phase 4 - Temporal Integrity Audit</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 11–12</a> — heading ID <code class="cw-code">phase-4-heading</code>, output cell 12 ID <code class="cw-code">phase-4-orchestration</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="cw-code">render_phase_summary(4, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/temporal/phase_4_signoff.json">phase_4_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 11 (markdown):</strong> Tiêu đề <code class="cw-code">## Phase 4 - Temporal Integrity Audit</code>
<strong class="cw-strong">Cell 12 (code):</strong> Chạy <code class="cw-code">render_phase_summary(4, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng của Cell 12:</strong></p><ul class="cw-list">
<li class="cw-li">Thống kê: cadence = 10 phút, no gaps, no duplicates, continuity segments.</li>
<li class="cw-li">File <code class="cw-code">artifacts/temporal/{temporal_manifest.json, daily_observation_counts.csv, interval_distribution.csv}</code>.</li>
</ul><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Verify time series đều đặn ở cadence 10 phút.</li>
<li class="cw-li">Nếu có gaps, sẽ có <code class="cw-code">continuity_segments.csv</code> liệt kê các đoạn liên tục.</li>
<li class="cw-li">Đây là gate quan trọng trước khi EDA.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 12 trong notebook</a> → tìm cell ID <code class="cw-code">phase-4-orchestration</code>.</p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 5 · NOTEBOOK WALKTHROUGH</span><a id="cell-13"></a>Cell 13–14 — Phase 5 - Chronological Split</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 13–14</a> — heading ID <code class="cw-code">phase-5-heading</code>, output cell 14 ID <code class="cw-code">phase-5-orchestration</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="cw-code">render_phase_summary(5, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/splits/phase_5_signoff.json">phase_5_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 13 (markdown):</strong> Tiêu đề <code class="cw-code">## Phase 5 - Chronological Split</code>
<strong class="cw-strong">Cell 14 (code):</strong> Chạy <code class="cw-code">render_phase_summary(5, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng của Cell 14:</strong></p><ul class="cw-list">
<li class="cw-li">Split statistics: Train 70% / Validation 15% / Test 15%.</li>
<li class="cw-li">File <code class="cw-code">artifacts/splits/{split_manifest.json, split_membership.csv, split_boundaries.csv, split_leakage_audit.csv}</code>.</li>
</ul><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Split chronological: KHÔNG shuffle (giữ nguyên thứ tự thời gian).</li>
<li class="cw-li">Leakage audit đảm bảo không có sample overlap giữa Train/Val/Test.</li>
<li class="cw-li">Boundaries xác định timestamp cắt: thường Train <code class="cw-code">≤ 2016-04-15</code>, Val <code class="cw-code">2016-04-15 → 2016-05-12</code>, Test <code class="cw-code">≥ 2016-05-12</code>.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 14 trong notebook</a> → tìm cell ID <code class="cw-code">phase-5-orchestration</code>.</p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 6 · NOTEBOOK WALKTHROUGH</span><a id="cell-15"></a>Cell 15–44 — Phase 6 - Exploratory Data Analysis</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 15–44</a> — heading ID <code class="cw-code">phase-6-heading</code>, output cell 16 ID <code class="cw-code">phase-6-setup</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="cw-code">render_phase_summary(6, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/eda/phase_6_signoff.json">phase_6_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 15 (markdown):</strong> Tiêu đề <code class="cw-code">## Phase 6 - Exploratory Data Analysis</code>
<strong class="cw-strong">Cell 16 (code):</strong> <code class="cw-code">from io import BytesIO</code>, đọc interim CSV</p><p class="cw-p"><strong class="cw-strong">Các sub-cells của Phase 6:</strong></p><div class="cw-table-wrap"><table class="cw-table">
<thead>
<tr>
<th>Sub-cell</th>
<th>Markdown Header</th>
<th>Code action</th>
<th>Output kỳ vọng</th>
</tr>
</thead>
<tbody>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 17</a></td>
<td><code class="cw-code">### 6.1 Data Overview</code></td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 18</a></td>
<td>—</td>
<td><code class="cw-code">df.head(8)</code></td>
<td>DataFrame 8 dòng đầu, 29 cột</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 19</a></td>
<td><code class="cw-code">### 6.2 Missing-Value Analysis</code></td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 20</a></td>
<td>—</td>
<td><code class="cw-code">df.isna().sum()</code></td>
<td>Series đếm missing mỗi cột (thường tất cả = 0)</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 21</a></td>
<td><code class="cw-code">### 6.3 Calendar Derivations</code></td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 22</a></td>
<td>—</td>
<td><code class="cw-code">df["hour"] = df["date"].dt.hour</code></td>
<td>Không output, side-effect</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 23</a></td>
<td><code class="cw-code">### 6.4 Target Distribution</code></td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 24</a></td>
<td>—</td>
<td>đọc <code class="cw-code">eda_numeric_summary.csv</code></td>
<td>Bảng summary target</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 25</a></td>
<td><code class="cw-code">### 6.5 Target Timeline</code></td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 26</a></td>
<td>—</td>
<td><code class="cw-code">df.set_index("date")</code></td>
<td>Setup cho plotting</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 27</a></td>
<td><code class="cw-code">### 6.6 Calendar Energy Patterns</code></td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 28</a></td>
<td>—</td>
<td>display CSV</td>
<td>Bảng hourly profile</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 29</a></td>
<td><code class="cw-code">### 6.7 Feature Distributions</code></td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 30</a></td>
<td>—</td>
<td><code class="cw-code">Image(filename=EDA_10_...)</code></td>
<td>Hình phân phối features</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 31</a></td>
<td><code class="cw-code">### 6.8 Numerical Relationships</code></td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 32</a></td>
<td>—</td>
<td>setup sensor_columns</td>
<td>List tên cột cảm biến</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 33</a></td>
<td><code class="cw-code">### 6.9 Cross-Correlation at Short Lags</code></td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 34</a></td>
<td>—</td>
<td>define cross-corr function</td>
<td>Không output</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 35</a></td>
<td><code class="cw-code">### 6.10 Categorical and Numerical</code></td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 36</a></td>
<td>—</td>
<td>groupby is_weekend</td>
<td>Describe table</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 37</a></td>
<td><code class="cw-code">### 6.11 IQR Outlier Diagnostics</code></td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 38</a></td>
<td>—</td>
<td>define IQR summary</td>
<td>Không output</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 39</a></td>
<td><code class="cw-code">### 6.12 Outlier-Smoothing Demo</code></td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 40</a></td>
<td>—</td>
<td>df.copy(deep=True)</td>
<td>Không output</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 41</a></td>
<td><code class="cw-code">### 6.13 Time-Series Diagnostics</code></td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 42</a></td>
<td>—</td>
<td>đọc lag correlations</td>
<td>Bảng lag correlation</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 43</a></td>
<td><code class="cw-code">### 6.14 Extreme Samples</code></td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cell 44</a></td>
<td>—</td>
<td>đọc extreme_target_samples</td>
<td>Bảng extreme samples</td>
</tr>
</tbody>
</table></div><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích Phase 6:</strong></p><ul class="cw-list">
<li class="cw-li">Phase dài nhất trong foundation (30 cells), mục đích khám phá dữ liệu toàn diện.</li>
<li class="cw-li">Kỳ vọng output: tables (CSV) + figures (PNG) được đọc/hiển thị inline.</li>
<li class="cw-li">Tất cả figures nằm trong <code class="cw-code">artifacts/eda/figures/</code>, tables trong <code class="cw-code">artifacts/eda/tables/</code>.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp output đầu tiên của Phase 6 tại Cell 18</a> → tìm cells từ <code class="cw-code">## Phase 6 - Exploratory Data Analysis</code> đến trước <code class="cw-code">## Phase 7</code></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 7 · NOTEBOOK WALKTHROUGH</span><a id="cell-45"></a>Cell 45–46 — Phase 7 - Feature Engineering</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 45–46</a> — heading ID <code class="cw-code">phase-7-heading</code>, output cell 46 ID <code class="cw-code">phase-7-orchestration</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="cw-code">render_phase_summary(7, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/features/phase_7_signoff.json">phase_7_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 45 (markdown):</strong> Tiêu đề <code class="cw-code">## Phase 7 - Feature Engineering</code>
<strong class="cw-strong">Cell 46 (code):</strong> Chạy <code class="cw-code">render_phase_summary(7, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng của Cell 46:</strong></p><ul class="cw-list">
<li class="cw-li">Manifest với feature list (time features, lag features, rolling stats).</li>
<li class="cw-li">File <code class="cw-code">data/interim/uci_appliances_energy_prediction/energydata_feature_engineered_v1.csv</code>.</li>
<li class="cw-li">File <code class="cw-code">artifacts/features/{feature_engineering_manifest.json, feature_registry.csv, feature_lineage.csv}</code>.</li>
</ul><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Engineering features chỉ dựa trên Train, KHÔNG leak từ Val/Test.</li>
<li class="cw-li">File <code class="cw-code">feature_engineered_v1.csv</code> có ~33 cột (29 gốc + 4 time + lag + rolling).</li>
<li class="cw-li">SHA256 file <code class="cw-code">feature_engineered_v1.sha256</code> đảm bảo determinism.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 46 trong notebook</a> → tìm cell ID <code class="cw-code">phase-7-orchestration</code>.</p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 8 · NOTEBOOK WALKTHROUGH</span><a id="cell-47"></a>Cell 47–48 — Phase 8 - Feature-Set Variants</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 47–48</a> — heading ID <code class="cw-code">phase-8-heading</code>, output cell 48 ID <code class="cw-code">phase-8-orchestration</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="cw-code">render_phase_summary(8, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/feature_sets/phase_8_signoff.json">phase_8_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 47 (markdown):</strong> Tiêu đề <code class="cw-code">## Phase 8 - Feature-Set Variants</code>
<strong class="cw-strong">Cell 48 (code):</strong> Chạy <code class="cw-code">render_phase_summary(8, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng của Cell 48:</strong></p><ul class="cw-list">
<li class="cw-li">Registry 3 variants: FS0, FS1, FS2.</li>
<li class="cw-li">File <code class="cw-code">artifacts/feature_sets/{feature_set_registry.json, feature_components.json, feature_set_lineage.csv}</code>.</li>
</ul><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">3 variants để sweep Phase 23 (S1).</li>
<li class="cw-li">FS0 = minimal, FS1 = + time, FS2 = + lag features.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 48 trong notebook</a> → tìm cell ID <code class="cw-code">phase-8-orchestration</code>.</p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 9 · NOTEBOOK WALKTHROUGH</span><a id="cell-49"></a>Cell 49–50 — Phase 9 - Train-Only Scaling</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 49–50</a> — heading ID <code class="cw-code">phase-9-heading</code>, output cell 50 ID <code class="cw-code">phase-9-orchestration</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="cw-code">render_phase_summary(9, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/scaling/phase_9_signoff.json">phase_9_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 49 (markdown):</strong> Tiêu đề <code class="cw-code">## Phase 9 - Train-Only Scaling</code>
<strong class="cw-strong">Cell 50 (code):</strong> Chạy <code class="cw-code">render_phase_summary(9, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng của Cell 50:</strong></p><ul class="cw-list">
<li class="cw-li">Scaler fingerprints (StandardScaler for X, StandardScaler for y).</li>
<li class="cw-li">File <code class="cw-code">artifacts/scalers/{x,y}/*.joblib</code>.</li>
<li class="cw-li">File <code class="cw-code">artifacts/scaling/{scaler_registry.json, scaling_audit.csv, scaling_discrepancies.json}</code>.</li>
</ul><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Scaler được FIT trên Train only.</li>
<li class="cw-li">Val và Test được TRANSFORM bằng scaler của Train → không leak.</li>
<li class="cw-li">Output thường chứa <code class="cw-code">scaler.mean_</code> và <code class="cw-code">scaler.scale_</code> per feature.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 50 trong notebook</a> → tìm cell ID <code class="cw-code">phase-9-orchestration</code>.</p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 10 · NOTEBOOK WALKTHROUGH</span><a id="cell-51"></a>Cell 51–52 — Phase 10 - Window Builder</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 51–52</a> — heading ID <code class="cw-code">phase-10-heading</code>, output cell 52 ID <code class="cw-code">phase-10-orchestration</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="cw-code">render_phase_summary(10, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/windows/phase_10_signoff.json">phase_10_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 51 (markdown):</strong> Tiêu đề <code class="cw-code">## Phase 10 - Window Builder</code>
<strong class="cw-strong">Cell 52 (code):</strong> Chạy <code class="cw-code">render_phase_summary(10, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng của Cell 52:</strong></p><ul class="cw-list">
<li class="cw-li">Window population summary.</li>
<li class="cw-li">File <code class="cw-code">artifacts/windows/{window_manifest.json, window_population_summary.csv, window_fingerprints.json}</code>.</li>
</ul><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Build (X, y) pairs với lookback ∈ {36, 72, 144}.</li>
<li class="cw-li">Population phải giảm dần: Train &gt; Val &gt; Test (do mỗi sample cần lookback).</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 52 trong notebook</a> → tìm cell ID <code class="cw-code">phase-10-orchestration</code>.</p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 11 · NOTEBOOK WALKTHROUGH</span><a id="cell-53"></a>Cell 53–54 — Phase 11 - DataLoaders</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 53–54</a> — heading ID <code class="cw-code">phase-11-heading</code>, output cell 54 ID <code class="cw-code">phase-11-orchestration</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="cw-code">render_phase_summary(11, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/dataloaders/phase_11_signoff.json">phase_11_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 53 (markdown):</strong> Tiêu đề <code class="cw-code">## Phase 11 - DataLoaders</code>
<strong class="cw-strong">Cell 54 (code):</strong> Chạy <code class="cw-code">render_phase_summary(11, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng của Cell 54:</strong></p><ul class="cw-list">
<li class="cw-li">DataLoader configs: batch_size, shuffle policy (Train=shuffle, Val/Test=no shuffle).</li>
<li class="cw-li">File <code class="cw-code">artifacts/dataloaders/{dataloader_manifest.json, dataloader_registry.csv, sequential_order_audit.csv}</code>.</li>
</ul><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Train DataLoader shuffle với seed cố định.</li>
<li class="cw-li">Val/Test DataLoader KHÔNG shuffle để giữ thứ tự thời gian.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 54 trong notebook</a> → tìm cell ID <code class="cw-code">phase-11-orchestration</code>.</p><hr class="cw-rule"/></article></section><section class="cw-section" data-section="5"><div class="cw-section-heading"><span class="cw-section-index">05</span><h2 class="cw-heading cw-h2">Giai Đoạn Modeling (Cells 55–76)</h2></div><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 12 · NOTEBOOK WALKTHROUGH</span><a id="cell-55"></a>Cell 55–56 — Phase 12 - Shared Metrics</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 55–56</a> — heading ID <code class="cw-code">phase-12-heading</code>, output cell 56 ID <code class="cw-code">phase-12-orchestration</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="cw-code">render_phase_summary(12, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/metrics/phase_12_signoff.json">phase_12_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 55 (markdown):</strong> Tiêu đề <code class="cw-code">## Phase 12 - Shared Metrics</code>
<strong class="cw-strong">Cell 56 (code):</strong> Chạy <code class="cw-code">render_phase_summary(12, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng của Cell 56:</strong></p><ul class="cw-list">
<li class="cw-li">Manifest với metrics: MAE, RMSE, R², MAPE (addendum).</li>
<li class="cw-li">File <code class="cw-code">artifacts/metrics/{metric_manifest.json, metric_unit_tests.csv, metric_reference_examples.csv}</code>.</li>
</ul><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Metrics tính ở đơn vị Wh gốc (inverse_transform y_pred, y_true).</li>
<li class="cw-li">R² ở Wh scale (không phải scaled space).</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 56 trong notebook</a> → tìm cell ID <code class="cw-code">phase-12-orchestration</code>.</p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 13 · NOTEBOOK WALKTHROUGH</span><a id="cell-57"></a>Cell 57–58 — Phase 13 - Experiment Registry</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 57–58</a> — heading ID <code class="cw-code">phase-13-heading</code>, output cell 58 ID <code class="cw-code">phase-13-orchestration</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="cw-code">render_phase_summary(13, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/experiments/phase_13_signoff.json">phase_13_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 57 (markdown):</strong> Tiêu đề <code class="cw-code">## Phase 13 - Experiment Registry</code>
<strong class="cw-strong">Cell 58 (code):</strong> Chạy <code class="cw-code">render_phase_summary(13, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng của Cell 58:</strong></p><ul class="cw-list">
<li class="cw-li">Empty registry (chưa có run nào).</li>
<li class="cw-li">File <code class="cw-code">artifacts/experiments/{experiment_registry.jsonl, registry_manifest.json, sweep_registry.csv}</code>.</li>
</ul><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Registry sẽ được fill dần khi các sweep phases chạy.</li>
<li class="cw-li">Mỗi training run được register với: id, config, status, metrics.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 58 trong notebook</a> → tìm cell ID <code class="cw-code">phase-13-orchestration</code>.</p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 14 · NOTEBOOK WALKTHROUGH</span><a id="cell-59"></a>Cell 59–60 — Phase 14 - Persistence Baseline</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 59–60</a> — heading ID <code class="cw-code">phase-14-heading</code>, output cell 60 ID <code class="cw-code">phase-14-orchestration</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="cw-code">render_phase_summary(14, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/baselines/persistence/phase_14_signoff.json">phase_14_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 59 (markdown):</strong> Tiêu đề <code class="cw-code">## Phase 14 - Persistence Baseline</code>
<strong class="cw-strong">Cell 60 (code):</strong> Chạy <code class="cw-code">render_phase_summary(14, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng của Cell 60:</strong></p><ul class="cw-list">
<li class="cw-li">Validation metrics cho persistence: MAE ≈ 26 Wh, RMSE ≈ 47 Wh, R² ≈ 0.20.</li>
<li class="cw-li">File <code class="cw-code">artifacts/baselines/persistence/{persistence_manifest.json, persistence_validation_metrics.json}</code>.</li>
</ul><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Persistence: $\hat{y}[t+1] = y[t]$ (giá trị step trước).</li>
<li class="cw-li">Đây là baseline yếu nhất, nhưng khó bị đánh bại ở những biến động nhỏ.</li>
<li class="cw-li">Nếu LSTM/Transformer không beat được → có vấn đề về features hoặc training.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 60 trong notebook</a> → tìm cell ID <code class="cw-code">phase-14-orchestration</code>.</p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 15 · NOTEBOOK WALKTHROUGH</span><a id="cell-61"></a>Cell 61–62 — Phase 15 - LSTM Implementation</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 61–62</a> — heading ID <code class="cw-code">1d1eebe5</code>, output cell 62 ID <code class="cw-code">499bc911</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="cw-code">render_phase_summary(15, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/models/lstm/phase_15_signoff.json">phase_15_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 61 (markdown):</strong> Tiêu đề <code class="cw-code">## Phase 15 - LSTM Implementation</code>
<strong class="cw-strong">Cell 62 (code):</strong> Chạy <code class="cw-code">render_phase_summary(15, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng của Cell 62:</strong></p><ul class="cw-list">
<li class="cw-li">LSTM architecture fingerprint: hidden_size, num_layers, dropout, total_params.</li>
<li class="cw-li">File <code class="cw-code">artifacts/models/lstm/{lstm_model_manifest.json, lstm_shape_contract.json, lstm_unit_tests.csv}</code>.</li>
</ul><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Implement LSTM regressor (KHÔNG train ở phase này).</li>
<li class="cw-li">Shape contract đảm bảo input <code class="cw-code">(batch, lookback, n_features)</code> → output <code class="cw-code">(batch, 1)</code>.</li>
<li class="cw-li">Unit tests verify shape correctness trên dummy input.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 62 trong notebook</a> → tìm cell ID <code class="cw-code">499bc911</code>.</p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 16 · NOTEBOOK WALKTHROUGH</span><a id="cell-63"></a>Cell 63–64 — Phase 16 - Transformer Implementation</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 63–64</a> — heading ID <code class="cw-code">d91e6b47</code>, output cell 64 ID <code class="cw-code">518fca61</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="cw-code">render_phase_summary(16, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/models/transformer/phase_16_signoff.json">phase_16_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 63 (markdown):</strong> Tiêu đề <code class="cw-code">## Phase 16 - Transformer Implementation</code>
<strong class="cw-strong">Cell 64 (code):</strong> Chạy <code class="cw-code">render_phase_summary(16, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng của Cell 64:</strong></p><ul class="cw-list">
<li class="cw-li">Transformer architecture fingerprint: d_model, num_heads, num_layers, ffn_dim, total_params.</li>
<li class="cw-li">File <code class="cw-code">artifacts/models/transformer/{transformer_model_manifest.json, transformer_attention_contract.json, transformer_shape_contract.json, ...}</code>.</li>
</ul><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Implement Transformer encoder regressor (KHÔNG train).</li>
<li class="cw-li">Attention contract: output attention shape <code class="cw-code">(batch, num_heads, seq, seq)</code>.</li>
<li class="cw-li">Positional encoding sin/cos được add vào input embedding.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 64 trong notebook</a> → tìm cell ID <code class="cw-code">518fca61</code>.</p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 17 · NOTEBOOK WALKTHROUGH</span><a id="cell-65"></a>Cell 65–66 — Phase 17 - Attention-Aware Encoder Verification</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 65–66</a> — heading ID <code class="cw-code">8906dc9f</code>, output cell 66 ID <code class="cw-code">efc7f487</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="cw-code">render_phase_summary(17, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/attention_verification/phase_17_signoff.json">phase_17_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 65 (markdown):</strong> Tiêu đề <code class="cw-code">## Phase 17 - Attention-Aware Encoder Verification</code>
<strong class="cw-strong">Cell 66 (code):</strong> Chạy <code class="cw-code">render_phase_summary(17, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng của Cell 66:</strong></p><ul class="cw-list">
<li class="cw-li">Audit reports: shape audit, probability audit (sum=1), mask audit, path equivalence.</li>
<li class="cw-li">File <code class="cw-code">artifacts/attention_verification/{attention_verification_manifest.json, attention_probability_audit.csv, attention_path_equivalence_audit.csv, ...}</code>.</li>
</ul><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Verify attention weights là valid probability distribution.</li>
<li class="cw-li">Path equivalence: attention từ manual computation == framework output.</li>
<li class="cw-li">Nếu FAIL ở phase này → Phase 18 sẽ không pass.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 66 trong notebook</a> → tìm cell ID <code class="cw-code">efc7f487</code>.</p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 18 · NOTEBOOK WALKTHROUGH</span><a id="cell-67"></a>Cell 67–68 — Phase 18 - Forward-Pass Sanity Tests</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 67–68</a> — heading ID <code class="cw-code">6dee6a62</code>, output cell 68 ID <code class="cw-code">d60e1951</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="cw-code">render_phase_summary(18, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/forward_sanity/phase_18_signoff.json">phase_18_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 67 (markdown):</strong> Tiêu đề <code class="cw-code">## Phase 18 - Forward-Pass Sanity Tests</code>
<strong class="cw-strong">Cell 68 (code):</strong> Chạy <code class="cw-code">render_phase_summary(18, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng của Cell 68:</strong></p><ul class="cw-list">
<li class="cw-li">Forward pass tests: batch independence, parameter non-mutation, device transfer, scaling correctness.</li>
<li class="cw-li">File <code class="cw-code">artifacts/forward_sanity/{forward_sanity_manifest.json, forward_batch_audit.csv, ...}</code>.</li>
</ul><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Sanity tests trên model đã initialized (chưa trained).</li>
<li class="cw-li">Đảm bảo forward pass deterministic và không side-effect.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 68 trong notebook</a> → tìm cell ID <code class="cw-code">d60e1951</code>.</p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 19 · NOTEBOOK WALKTHROUGH</span><a id="cell-69"></a>Cell 69–70 — Phase 19 - Baseline Training Engine</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 69–70</a> — heading ID <code class="cw-code">a5fd715f</code>, output cell 70 ID <code class="cw-code">2acc38f9</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="cw-code">render_phase_summary(19, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/training_engine/phase_19_signoff.json">phase_19_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 69 (markdown):</strong> Tiêu đề <code class="cw-code">## Phase 19 - Baseline Training Engine</code>
<strong class="cw-strong">Cell 70 (code):</strong> Chạy <code class="cw-code">render_phase_summary(19, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng của Cell 70:</strong></p><ul class="cw-list">
<li class="cw-li">Training engine contract: optimizer template, scheduler template, checkpoint schema.</li>
<li class="cw-li">File <code class="cw-code">artifacts/training_engine/{training_engine_manifest.json, training_engine_unit_tests.csv, checkpoint_schema.json}</code>.</li>
</ul><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Implement generic training loop (KHÔNG train model thật ở đây).</li>
<li class="cw-li">Engine hỗ trợ: gradient clipping, early stopping, checkpoint save/load.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 70 trong notebook</a> → tìm cell ID <code class="cw-code">2acc38f9</code>.</p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 20 · NOTEBOOK WALKTHROUGH</span><a id="cell-71"></a>Cell 71–72 — Phase 20 - LSTM Baseline Run</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 71–72</a> — heading ID <code class="cw-code">11dc2d67</code>, output cell 72 ID <code class="cw-code">2eb13c9c</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="cw-code">render_phase_summary(20, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/lstm_baseline/phase_20_signoff.json">phase_20_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 71 (markdown):</strong> Tiêu đề <code class="cw-code">## Phase 20 - LSTM Baseline Run</code>
<strong class="cw-strong">Cell 72 (code):</strong> Chạy <code class="cw-code">render_phase_summary(20, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng của Cell 72:</strong></p><ul class="cw-list">
<li class="cw-li">LSTM training log + final validation metrics.</li>
<li class="cw-li">File <code class="cw-code">artifacts/runs/RUN_LS_LS_*/{config.json, status.json, training.log, training_history.csv, metrics/best_validation_metrics.json}</code>.</li>
</ul><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Train LSTM baseline với reference config.</li>
<li class="cw-li">Validation RMSE thường ≈ 73 Wh (vs persistence 47 Wh → LSTM ban đầu chưa beat persistence vì hyperparameters chưa tuned).</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 72 trong notebook</a> → tìm cell ID <code class="cw-code">2eb13c9c</code>.</p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 21 · NOTEBOOK WALKTHROUGH</span><a id="cell-73"></a>Cell 73–74 — Phase 21 - Transformer B0 Run</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 73–74</a> — heading ID <code class="cw-code">328a4c95</code>, output cell 74 ID <code class="cw-code">03db9e36</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="cw-code">render_phase_summary(21, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/transformer_b0/phase_21_signoff.json">phase_21_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 73 (markdown):</strong> Tiêu đề <code class="cw-code">## Phase 21 - Transformer B0 Run</code>
<strong class="cw-strong">Cell 74 (code):</strong> Chạy <code class="cw-code">render_phase_summary(21, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng của Cell 74:</strong></p><ul class="cw-list">
<li class="cw-li">Transformer B0 training log + validation metrics.</li>
<li class="cw-li">File <code class="cw-code">artifacts/runs/RUN_TR_B0_*/</code> (config, status, history, metrics).</li>
</ul><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Train Transformer B0 (baseline config chưa tune).</li>
<li class="cw-li">Thường có RMSE ≈ 60-70 Wh.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 74 trong notebook</a> → tìm cell ID <code class="cw-code">03db9e36</code>.</p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 22 · NOTEBOOK WALKTHROUGH</span><a id="cell-75"></a>Cell 75–76 — Phase 22 - Learning-Curve Diagnostics</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 75–76</a> — heading ID <code class="cw-code">410cc5e0</code>, output cell 76 ID <code class="cw-code">cd4716af</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="cw-code">render_frozen_phase_evidence(22, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/learning_diagnostics/phase_22_signoff.json">phase_22_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 75 (markdown):</strong> Tiêu đề <code class="cw-code">## Phase 22 - Learning-Curve Diagnostics</code>
<strong class="cw-strong">Cell 76 (code):</strong> <code class="cw-code">render_phase_summary(...)</code> cho Phase 20/21</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng của Cell 76:</strong></p><ul class="cw-list">
<li class="cw-li">HTML dashboard hiển thị learning curves cho các baseline runs.</li>
<li class="cw-li">So sánh LSTM vs Transformer B0 loss curves.</li>
</ul><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Phase chẩn đoán: Train vs Validation loss có converge không?</li>
<li class="cw-li">Phát hiện overfitting/underfitting sớm để biết sweep nào cần thiết.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 76 trong notebook</a> → tìm cell chứa <code class="cw-code">render_phase_summary</code></p><hr class="cw-rule"/></article></section><section class="cw-section" data-section="6"><div class="cw-section-heading"><span class="cw-section-index">06</span><h2 class="cw-heading cw-h2">Giai Đoạn Sweeps (Cells 77–114)</h2></div><p class="cw-p">Mỗi sweep phase gồm 1 markdown + 1 code cell. Code cell sẽ:</p><ol class="cw-list cw-ordered">
<li class="cw-li">Load reference config từ phase trước.</li>
<li class="cw-li">Run sweep variants.</li>
<li class="cw-li">Pick winner bằng validation RMSE.</li>
<li class="cw-li">Update reference config.</li>
</ol><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 23 · NOTEBOOK WALKTHROUGH</span><a id="cell-77"></a>Cell 77–78 — Phase 23 - S1 Feature-Set Sweep</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 77–78</a> — heading ID <code class="cw-code">08a52851</code>, output cell 78 ID <code class="cw-code">0881cfe3</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="cw-code">render_frozen_phase_evidence(23, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/sweeps/s1_feature_set/phase_23_signoff.json">phase_23_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 77:</strong> Markdown tiêu đề <code class="cw-code">## Phase 23 - S1 Feature-Set Sweep</code>
<strong class="cw-strong">Cell 78:</strong> <code class="cw-code">render_frozen_phase_evidence(23, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng:</strong> Bảng CSV <code class="cw-code">results.csv</code> với các variants FS0, FS1, FS2 và validation metrics. Winner = best validation RMSE.</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Sweep 3 feature sets.</li>
<li class="cw-li">Update reference → Phase 24 sẽ dùng winner.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 78 trong notebook</a> → tìm cell có <code class="cw-code">Phase 23 - S1 Feature-Set Sweep</code></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 24 · NOTEBOOK WALKTHROUGH</span><a id="cell-79"></a>Cell 79–80 — Phase 24 - S2 Time-Feature Sweep</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 79–80</a> — heading ID <code class="cw-code">35e8f752</code>, output cell 80 ID <code class="cw-code">9b2d6e88</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="cw-code">render_frozen_phase_evidence(24, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/sweeps/s2_time_feature/phase_24_signoff.json">phase_24_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 79:</strong> Markdown tiêu đề
<strong class="cw-strong">Cell 80:</strong> <code class="cw-code">render_frozen_phase_evidence(24, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng:</strong> Sweep time features (hour sin/cos, dayofweek, is_weekend, ...).</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 80 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 25 · NOTEBOOK WALKTHROUGH</span><a id="cell-81"></a>Cell 81–82 — Phase 25 - S3 Target-Scaling Sweep</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 81–82</a> — heading ID <code class="cw-code">1a1d6b45</code>, output cell 82 ID <code class="cw-code">51bc5965</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="cw-code">render_frozen_phase_evidence(25, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/sweeps/s3_target_scaling/phase_25_signoff.json">phase_25_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
Sweep target scaling: StandardScaler, RobustScaler, MinMaxScaler, no-scaling.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 82 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 26 · NOTEBOOK WALKTHROUGH</span><a id="cell-83"></a>Cell 83–84 — Phase 26 - S4 Lookback Sweep</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 83–84</a> — heading ID <code class="cw-code">2ef1b3d9</code>, output cell 84 ID <code class="cw-code">585b390d</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="cw-code">render_frozen_phase_evidence(26, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/sweeps/s4_lookback/phase_26_signoff.json">phase_26_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
Sweep lookback ∈ {36, 72, 144}.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 84 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 27 · NOTEBOOK WALKTHROUGH</span><a id="cell-85"></a>Cell 85–86 — Phase 27 - S5 Pooling Sweep</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 85–86</a> — heading ID <code class="cw-code">2b25f489</code>, output cell 86 ID <code class="cw-code">86f4ac0c</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="cw-code">render_frozen_phase_evidence(27, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/sweeps/s5_pooling/phase_27_signoff.json">phase_27_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
Sweep pooling: last, mean, max, attention.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 86 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 28 · NOTEBOOK WALKTHROUGH</span><a id="cell-87"></a>Cell 87–88 — Phase 28 - S6 Activation Sweep</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 87–88</a> — heading ID <code class="cw-code">b020e51d</code>, output cell 88 ID <code class="cw-code">b70c9707</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="cw-code">render_frozen_phase_evidence(28, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/sweeps/s6_activation/phase_28_signoff.json">phase_28_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
Sweep activation: ReLU, GELU, SiLU.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 88 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 29 · NOTEBOOK WALKTHROUGH</span><a id="cell-89"></a>Cell 89–90 — Phase 29 - S7 Batch-Size Sweep</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 89–90</a> — heading ID <code class="cw-code">aa9fb435</code>, output cell 90 ID <code class="cw-code">eb4f1b80</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="cw-code">render_frozen_phase_evidence(29, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/sweeps/s7_batch_size/phase_29_signoff.json">phase_29_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
Sweep batch_size ∈ {16, 32, 64, 128, 256}.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 90 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 30 · NOTEBOOK WALKTHROUGH</span><a id="cell-91"></a>Cell 91–92 — Phase 30 - S8 Learning-Rate Sweep</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 91–92</a> — heading ID <code class="cw-code">4a8618de</code>, output cell 92 ID <code class="cw-code">3fe4478c</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="cw-code">render_frozen_phase_evidence(30, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/sweeps/s8_learning_rate/phase_30_signoff.json">phase_30_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
Sweep learning_rate ∈ {1e-4, 5e-4, 1e-3, 5e-3}.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 92 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 31 · NOTEBOOK WALKTHROUGH</span><a id="cell-93"></a>Cell 93–94 — Phase 31 - S9 Weight-Decay Sweep</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 93–94</a> — heading ID <code class="cw-code">phase-31-heading</code>, output cell 94 ID <code class="cw-code">phase-31-resume</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="cw-code">render_frozen_phase_evidence(31, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/sweeps/S9_weight_decay/phase_31_signoff.json">phase_31_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 93:</strong> <code class="cw-code">phase-31-heading</code> ID
<strong class="cw-strong">Cell 94:</strong> <code class="cw-code">render_frozen_phase_evidence(31, PROJECT_ROOT)</code></p><p class="cw-p">Sweep weight_decay ∈ {0, 1e-5, 1e-4, 1e-3}.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 94 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 32 · NOTEBOOK WALKTHROUGH</span><a id="cell-95"></a>Cell 95–96 — Phase 32 - S10 Dropout Sweep</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 95–96</a> — heading ID <code class="cw-code">phase-32-heading</code>, output cell 96 ID <code class="cw-code">phase-32-resume</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="cw-code">render_frozen_phase_evidence(32, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/sweeps/S10_dropout/phase_32_signoff.json">phase_32_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 95:</strong> <code class="cw-code">phase-32-heading</code> ID
<strong class="cw-strong">Cell 96:</strong> <code class="cw-code">render_frozen_phase_evidence(32, PROJECT_ROOT)</code></p><p class="cw-p">Sweep dropout ∈ {0.0, 0.1, 0.2, 0.3}.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 96 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 33 · NOTEBOOK WALKTHROUGH</span><a id="cell-97"></a>Cell 97–98 — Transformer Configuration after Phase 33</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 97–98</a> — heading ID <code class="cw-code">phase-33-config-heading</code>, output cell 98 ID <code class="cw-code">phase-33-config-display</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="cw-code">render_phase_33_transformer_configuration()</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/sweeps/S11_d_model/phase_33_signoff.json">phase_33_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 97:</strong> Markdown tiêu đề <code class="cw-code">## Transformer Configuration after Phase 33</code>
<strong class="cw-strong">Cell 98:</strong> <code class="cw-code">render_phase_33_transformer_configuration(...)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng:</strong> Dashboard hiển thị toàn bộ Transformer config đã được update qua S1-S11 sweeps.</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Snapshot Transformer config TRƯỚC khi bắt đầu architecture sweeps (S12-S19).</li>
<li class="cw-li">Config này sẽ được dùng làm baseline cho S12-S19.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 98 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 34 · NOTEBOOK WALKTHROUGH</span><a id="cell-99"></a>Cell 99–100 — Phase 34 - S12 Head Sweep</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 99–100</a> — heading ID <code class="cw-code">phase-34-heading</code>, output cell 100 ID <code class="cw-code">phase-34-resume</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="cw-code">render_frozen_phase_evidence(34, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/sweeps/S12_heads/phase_34_signoff.json">phase_34_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1 RECOVERED</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.</p><p class="cw-p"><strong class="cw-strong">Trạng thái phục hồi:</strong> <code class="cw-code">VALID_REUSABLE</code>; action <code class="cw-code">RENDER_ONLY</code>. Reporting debt được giữ nguyên, không fabricate historical output.
<strong class="cw-strong">Cell 99:</strong> <code class="cw-code">phase-34-heading</code> ID
<strong class="cw-strong">Cell 100:</strong> <code class="cw-code">render_frozen_phase_evidence(34, PROJECT_ROOT)</code></p><p class="cw-p">Sweep <code class="cw-code">num_heads ∈ {2, 4, 8, 16}</code>.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 100 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 35 · NOTEBOOK WALKTHROUGH</span><a id="cell-101"></a>Cell 101–102 — Phase 35 - S13 Layer Sweep</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 101–102</a> — heading ID <code class="cw-code">phase-35-heading</code>, output cell 102 ID <code class="cw-code">phase-35-resume</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="cw-code">render_frozen_phase_evidence(35, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/sweeps/S13_layers/phase_35_signoff.json">phase_35_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1 RECOVERED</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.</p><p class="cw-p"><strong class="cw-strong">Trạng thái phục hồi:</strong> <code class="cw-code">VALID_REUSABLE</code>; action <code class="cw-code">RENDER_ONLY</code>. Reporting debt được giữ nguyên, không fabricate historical output.
Sweep <code class="cw-code">num_layers ∈ {1, 2, 3, 4, 6}</code>.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 102 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 36 · NOTEBOOK WALKTHROUGH</span><a id="cell-103"></a>Cell 103–104 — Phase 36 - S14 FFN Sweep</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 103–104</a> — heading ID <code class="cw-code">phase-36-heading</code>, output cell 104 ID <code class="cw-code">phase-36-resume</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="cw-code">render_frozen_phase_evidence(36, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/sweeps/S14_ffn/phase_36_signoff.json">phase_36_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1 RECOVERED</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.</p><p class="cw-p"><strong class="cw-strong">Trạng thái phục hồi:</strong> <code class="cw-code">VALID_REUSABLE</code>; action <code class="cw-code">RENDER_ONLY</code>. Reporting debt được giữ nguyên, không fabricate historical output.
Sweep FFN width multipliers ∈ {1, 2, 4}.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 104 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 37 · NOTEBOOK WALKTHROUGH</span><a id="cell-105"></a>Cell 105–106 — Phase 37 - S15 Loss Sweep</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 105–106</a> — heading ID <code class="cw-code">phase-37-heading</code>, output cell 106 ID <code class="cw-code">phase-37-resume</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="cw-code">render_frozen_phase_evidence(37, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/sweeps/S15_loss/phase_37_signoff.json">phase_37_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1 RECOVERED</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.</p><p class="cw-p"><strong class="cw-strong">Trạng thái phục hồi:</strong> <code class="cw-code">VALID_REUSABLE</code>; action <code class="cw-code">RENDER_ONLY</code>. Reporting debt được giữ nguyên, không fabricate historical output.
Sweep loss: MSE, Huber, Smooth L1.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 106 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 38 · NOTEBOOK WALKTHROUGH</span><a id="cell-107"></a>Cell 107–108 — Phase 38 - S16 Epoch-Cap Sweep</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 107–108</a> — heading ID <code class="cw-code">b3d754ee</code>, output cell 108 ID <code class="cw-code">17dc5a66</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="cw-code">render_frozen_phase_evidence(38, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/sweeps/S16_epoch_cap/phase_38_signoff.json">phase_38_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1 RECOVERED</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.</p><p class="cw-p"><strong class="cw-strong">Trạng thái phục hồi:</strong> <code class="cw-code">VALID_REUSABLE</code>; action <code class="cw-code">RENDER_ONLY</code>. Reporting debt được giữ nguyên, không fabricate historical output.
Sweep max_epochs ∈ {30, 50, 80, 100}.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 108 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 39 · NOTEBOOK WALKTHROUGH</span><a id="cell-109"></a>Cell 109–110 — Phase 39 - S17 Gradient-Clipping Sweep</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 109–110</a> — heading ID <code class="cw-code">d0833283</code>, output cell 110 ID <code class="cw-code">f0fcdce7</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="cw-code">render_frozen_phase_evidence(39, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/sweeps/S17_gradient_clipping/phase_39_signoff.json">phase_39_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1 RECOVERED</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.</p><p class="cw-p"><strong class="cw-strong">Trạng thái phục hồi:</strong> <code class="cw-code">VALID_REUSABLE</code>; action <code class="cw-code">RENDER_ONLY</code>. Reporting debt được giữ nguyên, không fabricate historical output.
Sweep gradient_clip ∈ {0.5, 1.0, 5.0, no_clip}.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 110 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 40 · NOTEBOOK WALKTHROUGH</span><a id="cell-111"></a>Cell 111–112 — Phase 40 - S18 RevIN Sweep</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 111–112</a> — heading ID <code class="cw-code">8a7873a9</code>, output cell 112 ID <code class="cw-code">d407f95c</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="cw-code">render_frozen_phase_evidence(40, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/sweeps/S18_revin/phase_40_signoff.json">phase_40_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1 RECOVERED</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.</p><p class="cw-p"><strong class="cw-strong">Trạng thái phục hồi:</strong> <code class="cw-code">VALID_REUSABLE</code>; action <code class="cw-code">RENDER_ONLY</code>. Reporting debt được giữ nguyên, không fabricate historical output.
Sweep RevIN on/off.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 112 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 41 · NOTEBOOK WALKTHROUGH</span><a id="cell-113"></a>Cell 113–114 — Phase 41 - S19 Boundary-Protocol Check</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 113–114</a> — heading ID <code class="cw-code">5acc48a2</code>, output cell 114 ID <code class="cw-code">7e86663d</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/frozen_evidence.py">frozen_evidence.py</a> — <code class="cw-code">render_frozen_phase_evidence(41, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/sweeps/S19_boundary_protocol/phase_41_signoff.json">phase_41_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1 RECOVERED</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.</p><p class="cw-p"><strong class="cw-strong">Trạng thái phục hồi:</strong> <code class="cw-code">VALID_REUSABLE</code>; action <code class="cw-code">RENDER_ONLY</code>. Reporting debt được giữ nguyên, không fabricate historical output.
Test boundary handling.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 114 trong notebook</a></p><hr class="cw-rule"/></article></section><section class="cw-section cw-section-final" data-section="7"><div class="cw-section-heading"><span class="cw-section-index">07</span><h2 class="cw-heading cw-h2">Giai Đoạn Final Pipeline và Benchmark Presentation (Cells 115–126)</h2></div><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 42 · NOTEBOOK WALKTHROUGH</span><a id="cell-115"></a>Cell 115–116 — Phase 42 - Candidate Synthesis</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 115–116</a> — heading ID <code class="cw-code">fa11ad6a</code>, output cell 116 ID <code class="cw-code">994bbdb9</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/phase_summary.py">phase_summary.py</a> — <code class="cw-code">render_phase_summary(42, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/candidate_synthesis/phase_42_signoff.json">phase_42_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 115:</strong> Markdown <code class="cw-code">## Phase 42 - Candidate Synthesis</code>
<strong class="cw-strong">Cell 116:</strong> <code class="cw-code">render_phase_summary(42, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng:</strong></p><ul class="cw-list">
<li class="cw-li">Transformer candidate shortlist + selected lineage.</li>
<li class="cw-li">File <code class="cw-code">artifacts/candidate_synthesis/{candidate_synthesis_manifest.json, transformer_candidate_shortlist.json, selected_lineage.json, baseline_anchor_context.json, boundary_sensitivity_context.json, candidate_synthesis_report.md}</code>.</li>
</ul><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Tổng hợp toàn bộ sweep winners.</li>
<li class="cw-li">Chọn candidate để đưa vào rolling origin (Phase 44).</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 116 trong notebook</a> → tìm cell ID <code class="cw-code">994bbdb9</code>.</p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 43 · NOTEBOOK WALKTHROUGH</span><a id="cell-117"></a>Cell 117–118 — Phase 43 — LSTM Tuning Results</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 117–118</a> — heading ID <code class="cw-code">ed1e48a7</code>, output cell 118 ID <code class="cw-code">6a082d01</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/results_rebuild.py">results_rebuild.py</a> — <code class="cw-code">render_verified_phase_result(43, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/lstm_tuning/phase_43_signoff.json">phase_43_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 117:</strong> Markdown <code class="cw-code">## Phase 43 - LSTM Tuning</code>
<strong class="cw-strong">Cell 118:</strong> <code class="cw-code">render_verified_phase_result(43, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng:</strong> Dashboard LSTM tuning stages (lt1-lt5).</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Tune LSTM hyperparameters theo stages.</li>
<li class="cw-li">Mỗi stage sweep 1 hyperparameter.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 118 trong notebook</a> → tìm cell chứa <code class="cw-code">render_verified_phase_result</code></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 44 · NOTEBOOK WALKTHROUGH</span><a id="cell-119"></a>Cell 119–120 — Phase 44 — Rolling-Origin Robustness Results</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 119–120</a> — heading ID <code class="cw-code">9eacc9e3</code>, output cell 120 ID <code class="cw-code">2f0130ea</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/results_rebuild.py">results_rebuild.py</a> — <code class="cw-code">render_verified_phase_result(44, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/rolling_origin/phase_44_signoff.json">phase_44_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 119:</strong> Markdown
<strong class="cw-strong">Cell 120:</strong> <code class="cw-code">render_verified_phase_result(44, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng:</strong> Dashboard rolling origin folds (5 folds × candidates).</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Robustness evaluation qua time slices.</li>
<li class="cw-li">Robust Lane: full refit mỗi fold.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 120 trong notebook</a> → tìm cell chứa <code class="cw-code">render_verified_phase_result</code></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 45 · NOTEBOOK WALKTHROUGH</span><a id="cell-121"></a>Cell 121–122 — Phase 45 — Final Model Lock Results</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 121–122</a> — heading ID <code class="cw-code">88d07113</code>, output cell 122 ID <code class="cw-code">4e6b273d</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/results_rebuild.py">results_rebuild.py</a> — <code class="cw-code">render_verified_phase_result(45, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/final_model_lock/final_model_lock_summary.json">final_model_lock_summary.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 121:</strong> Markdown <code class="cw-code">## Phase 45 - Final Model Lock</code>
<strong class="cw-strong">Cell 122:</strong> <code class="cw-code">render_verified_phase_result(45, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng:</strong> Dashboard locked config (architecture, optimizer, loss, scaler, ...).</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li"><strong class="cw-strong">NO-TRAIN phase</strong> — chỉ lock config.</li>
<li class="cw-li">Sau khi lock, không được thay đổi gì.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 122 trong notebook</a> → tìm cell chứa <code class="cw-code">render_verified_phase_result</code></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 46 · NOTEBOOK WALKTHROUGH</span><a id="cell-123"></a>Cell 123–124 — Phase 46 — Three-Seed Final Run Results</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 123–124</a> — heading ID <code class="cw-code">b388dcd1</code>, output cell 124 ID <code class="cw-code">753ab0f1</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/results_rebuild.py">results_rebuild.py</a> — <code class="cw-code">render_verified_phase_result(46, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/three_seed_final_runs/phase_46_signoff.json">phase_46_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Chế độ notebook hiện tại:</strong> Presentation-only; cell đọc artifact đã có, không chạy lại scientific computation.
<strong class="cw-strong">Cell 123:</strong> Markdown
<strong class="cw-strong">Cell 124:</strong> <code class="cw-code">render_verified_phase_result(46, PROJECT_ROOT)</code></p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Output kỳ vọng:</strong> Dashboard 3-seed runs (42, 123, 2026).</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Train final model với 3 seeds.</li>
<li class="cw-li">Mỗi seed tạo FINAL_REFIT checkpoint.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 124 trong notebook</a> → tìm cell chứa <code class="cw-code">render_verified_phase_result</code></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 47 · NOTEBOOK WALKTHROUGH</span><a id="cell-125"></a>Cell 125–126 — Phase 47 — Final V2 Post-hoc Benchmark Results</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 125–126</a> — heading ID <code class="cw-code">62f722b6</code>, output cell 126 ID <code class="cw-code">4dea0c14</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/results_rebuild.py">results_rebuild.py</a> — <code class="cw-code">render_verified_phase_result(47, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics_with_mape.json">step17_metrics_with_mape.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V2 FINAL</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Cell 125 (markdown) ID:</strong> <code class="cw-code">62f722b6</code>
<strong class="cw-strong">Cell 126 (code) ID:</strong> <code class="cw-code">4dea0c14</code>
<strong class="cw-strong">Renderer:</strong> <code class="cw-code">render_verified_phase_result(47, PROJECT_ROOT)</code>
<strong class="cw-strong">Output count kỳ vọng:</strong> <code class="cw-code">1</code></p><p class="cw-p"><strong class="cw-strong">Output:</strong> Bảng benchmark cho <code class="cw-code">V2 Seed 42</code>, <code class="cw-code">V2 Seed 123</code>, <code class="cw-code">V2 Seed 2026</code>, <code class="cw-code">V2 Equal-weight Ensemble</code> và <code class="cw-code">Persistence</code>, gồm <code class="cw-code">RMSE</code>, <code class="cw-code">MAE</code>, <code class="cw-code">MAPE</code> và <code class="cw-code">R²</code>.</p><div class="cw-table-wrap"><table class="cw-table">
<thead>
<tr>
<th>Model / seed</th>
<th style="text-align:right">RMSE Wh</th>
<th style="text-align:right">MAE Wh</th>
<th style="text-align:right">MAPE %</th>
<th style="text-align:right">R²</th>
<th>Decision</th>
</tr>
</thead>
<tbody>
<tr>
<td>V2 Seed 42</td>
<td style="text-align:right">61.262482</td>
<td style="text-align:right">27.292729</td>
<td style="text-align:right">24.187279</td>
<td style="text-align:right">0.545519</td>
<td>Component seed</td>
</tr>
<tr>
<td>V2 Seed 123</td>
<td style="text-align:right">63.259959</td>
<td style="text-align:right">27.332280</td>
<td style="text-align:right">23.147923</td>
<td style="text-align:right">0.515398</td>
<td>Component seed</td>
</tr>
<tr>
<td>V2 Seed 2026</td>
<td style="text-align:right">62.423633</td>
<td style="text-align:right">25.734231</td>
<td style="text-align:right">20.222825</td>
<td style="text-align:right">0.528127</td>
<td>Component seed</td>
</tr>
<tr>
<td>V2 Equal-weight Ensemble</td>
<td style="text-align:right">61.608937</td>
<td style="text-align:right">25.897130</td>
<td style="text-align:right">21.519219</td>
<td style="text-align:right">0.540364</td>
<td><code class="cw-code">FINAL_LOCKED_POLICY</code></td>
</tr>
<tr>
<td>Persistence</td>
<td style="text-align:right">66.836915</td>
<td style="text-align:right">26.737589</td>
<td style="text-align:right">21.513353</td>
<td style="text-align:right">0.459047</td>
<td>Baseline</td>
</tr>
</tbody>
</table></div><p class="cw-p cw-semantic-note"><strong class="cw-strong">Giải thích:</strong></p><ul class="cw-list">
<li class="cw-li">Đây là <code class="cw-code">POST_HOC_V2_BENCHMARK</code> của policy đã khóa; không phải unseen unbiased Test.</li>
<li class="cw-li">MAPE được đọc từ verified addendum artifact, không tính lại trong notebook.</li>
<li class="cw-li">Historical V1 Phase 47 artifacts vẫn được giữ riêng cho provenance.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở Phase 47 trong notebook</a></p><hr class="cw-rule"/></article></section><section class="cw-section cw-section-analysis" data-section="8"><div class="cw-section-heading"><span class="cw-section-index">08</span><h2 class="cw-heading cw-h2">Giai Đoạn Analysis (Cells 127–150)</h2></div><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 48 · NOTEBOOK WALKTHROUGH</span><a id="cell-127"></a>Cell 127–128 — Phase 48 — Final V2 Prediction Analysis Results</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 127–128</a> — heading ID <code class="cw-code">b496ac6f</code>, output cell 128 ID <code class="cw-code">1451ccf0</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/results_rebuild.py">results_rebuild.py</a> — <code class="cw-code">render_verified_phase_result(48, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/model_improvement_v2/final_reporting_analysis/phase48_prediction_analysis.json">phase48_prediction_analysis.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V2 FINAL</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Cell 127 (markdown) ID:</strong> <code class="cw-code">b496ac6f</code>
<strong class="cw-strong">Cell 128 (code) ID:</strong> <code class="cw-code">1451ccf0</code>
<strong class="cw-strong">Renderer:</strong> <code class="cw-code">render_verified_phase_result(48, PROJECT_ROOT)</code>
<strong class="cw-strong">Output count kỳ vọng:</strong> <code class="cw-code">1</code></p><p class="cw-p"><strong class="cw-strong">Output:</strong> Phân tích distribution, change magnitude, peak behavior và so sánh <code class="cw-code">V2 Equal-weight Ensemble</code> với <code class="cw-code">Persistence</code> trên 2.961 frozen prediction rows.</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Ý nghĩa:</strong> Đánh giá hình dạng prediction và các vùng biến động/peak mà không chạy model inference mới.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở Phase 48 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 49 · NOTEBOOK WALKTHROUGH</span><a id="cell-129"></a>Cell 129–130 — Phase 49 — Final V2 Residual Analysis Results</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 129–130</a> — heading ID <code class="cw-code">7cc29214</code>, output cell 130 ID <code class="cw-code">bbcf08f2</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/results_rebuild.py">results_rebuild.py</a> — <code class="cw-code">render_verified_phase_result(49, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/model_improvement_v2/final_reporting_analysis/phase49_residual_analysis.json">phase49_residual_analysis.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V2 FINAL</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Cell 129 (markdown) ID:</strong> <code class="cw-code">7cc29214</code>
<strong class="cw-strong">Cell 130 (code) ID:</strong> <code class="cw-code">bbcf08f2</code>
<strong class="cw-strong">Renderer:</strong> <code class="cw-code">render_verified_phase_result(49, PROJECT_ROOT)</code>
<strong class="cw-strong">Output count kỳ vọng:</strong> <code class="cw-code">1</code></p><p class="cw-p"><strong class="cw-strong">Output:</strong> Phân tích residual theo convention <code class="cw-code">y_true_wh - y_pred_wh</code>, signed bias và error distribution cho final ensemble và Persistence.</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Ý nghĩa:</strong> Kiểm tra bias còn lại trong frozen Step 17 predictions; đây là reporting-only analysis.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở Phase 49 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 50 · NOTEBOOK WALKTHROUGH</span><a id="cell-131"></a>Cell 131–132 — Phase 50 — Final V2 Error-by-Regime Analysis Results</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 131–132</a> — heading ID <code class="cw-code">9a2f1b89</code>, output cell 132 ID <code class="cw-code">1d5b5327</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/results_rebuild.py">results_rebuild.py</a> — <code class="cw-code">render_verified_phase_result(50, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/model_improvement_v2/final_reporting_analysis/phase50_error_by_regime.json">phase50_error_by_regime.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V2 FINAL</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Cell 131 (markdown) ID:</strong> <code class="cw-code">9a2f1b89</code>
<strong class="cw-strong">Cell 132 (code) ID:</strong> <code class="cw-code">1d5b5327</code>
<strong class="cw-strong">Renderer:</strong> <code class="cw-code">render_verified_phase_result(50, PROJECT_ROOT)</code>
<strong class="cw-strong">Output count kỳ vọng:</strong> <code class="cw-code">1</code></p><p class="cw-p"><strong class="cw-strong">Output:</strong> Error-by-regime dùng train-defined thresholds đã xác minh; <code class="cw-code">population_equality=PASS</code> và không dùng Test để fit threshold.</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Ý nghĩa:</strong> Tách hiệu năng theo regime mà không thay đổi final policy hoặc population.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở Phase 50 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 51 · NOTEBOOK WALKTHROUGH</span><a id="cell-133"></a>Cell 133–134 — Phase 51 — Final V2 Worst-Error Analysis Results</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 133–134</a> — heading ID <code class="cw-code">2e8f083e</code>, output cell 134 ID <code class="cw-code">c9df346a</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/results_rebuild.py">results_rebuild.py</a> — <code class="cw-code">render_verified_phase_result(51, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/model_improvement_v2/final_reporting_analysis/phase51_worst_error_analysis.json">phase51_worst_error_analysis.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V2 FINAL</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Cell 133 (markdown) ID:</strong> <code class="cw-code">2e8f083e</code>
<strong class="cw-strong">Cell 134 (code) ID:</strong> <code class="cw-code">c9df346a</code>
<strong class="cw-strong">Renderer:</strong> <code class="cw-code">render_verified_phase_result(51, PROJECT_ROOT)</code>
<strong class="cw-strong">Output count kỳ vọng:</strong> <code class="cw-code">1</code></p><p class="cw-p"><strong class="cw-strong">Output:</strong> Top-20 worst-error ranking của final ensemble, kèm case summary và tỷ trọng <code class="cw-code">SAE</code>/<code class="cw-code">SSE</code> từ frozen predictions.</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Ý nghĩa:</strong> Phân tích trường hợp khó nhất; không chọn seed và không dùng V1 case context thay cho V2.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở Phase 51 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 52 · NOTEBOOK WALKTHROUGH</span><a id="cell-135"></a>Cell 135–136 — Phase 52 — Historical V1 Attention Extraction Results</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 135–136</a> — heading ID <code class="cw-code">870efbe4</code>, output cell 136 ID <code class="cw-code">7b48186f</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/results_rebuild.py">results_rebuild.py</a> — <code class="cw-code">render_verified_phase_result(52, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/attention_extraction/attention_extraction_summary.json">attention_extraction_summary.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1 HISTORICAL ATTENTION</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Phạm vi khoa học:</strong> <code class="cw-code">V1 HISTORICAL ATTENTION</code>. Nội dung này không phải attention evidence của final V2 vì V2 Test attention tensors không được tạo.
<strong class="cw-strong">Cell 135 (markdown) ID:</strong> <code class="cw-code">870efbe4</code>
<strong class="cw-strong">Cell 136 (code) ID:</strong> <code class="cw-code">7b48186f</code>
<strong class="cw-strong">Source token:</strong> <code class="cw-code">render_verified_phase_result</code>
<strong class="cw-strong">Output count:</strong> <code class="cw-code">1</code></p><p class="cw-p"><strong class="cw-strong">Output:</strong> Hiển thị extraction state và checksum của raw attention artifacts.</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Ý nghĩa:</strong> Raw attention được materialize để các phase sau chỉ đọc lại.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 136 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 53 · NOTEBOOK WALKTHROUGH</span><a id="cell-137"></a>Cell 137–138 — Phase 53 — Historical V1 Attention Heatmap Results</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 137–138</a> — heading ID <code class="cw-code">afeb48a2</code>, output cell 138 ID <code class="cw-code">1c098fa6</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/results_rebuild.py">results_rebuild.py</a> — <code class="cw-code">render_verified_phase_result(53, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/attention_heatmaps/phase_53_signoff.json">phase_53_signoff.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1 HISTORICAL ATTENTION</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Phạm vi khoa học:</strong> <code class="cw-code">V1 HISTORICAL ATTENTION</code>. Nội dung này không phải attention evidence của final V2 vì V2 Test attention tensors không được tạo.
<strong class="cw-strong">Cell 137 (markdown) ID:</strong> <code class="cw-code">afeb48a2</code>
<strong class="cw-strong">Cell 138 (code) ID:</strong> <code class="cw-code">1c098fa6</code>
<strong class="cw-strong">Source token:</strong> <code class="cw-code">render_verified_phase_result</code>
<strong class="cw-strong">Output count:</strong> <code class="cw-code">1</code></p><p class="cw-p"><strong class="cw-strong">Output:</strong> Hiển thị heatmap theo case, seed, layer và head.</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Ý nghĩa:</strong> Cho phép so sánh attention distribution giữa các trường hợp và checkpoint.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 138 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 54 · NOTEBOOK WALKTHROUGH</span><a id="cell-139"></a>Cell 139–140 — Phase 54 — Historical V1 Last-Query Attention Results</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 139–140</a> — heading ID <code class="cw-code">3c915fcc</code>, output cell 140 ID <code class="cw-code">ec7b2ea5</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/results_rebuild.py">results_rebuild.py</a> — <code class="cw-code">render_verified_phase_result(54, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/last_query_attention/last_query_attention_summary.json">last_query_attention_summary.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1 HISTORICAL ATTENTION</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Phạm vi khoa học:</strong> <code class="cw-code">V1 HISTORICAL ATTENTION</code>. Nội dung này không phải attention evidence của final V2 vì V2 Test attention tensors không được tạo.
<strong class="cw-strong">Cell 139 (markdown) ID:</strong> <code class="cw-code">3c915fcc</code>
<strong class="cw-strong">Cell 140 (code) ID:</strong> <code class="cw-code">ec7b2ea5</code>
<strong class="cw-strong">Source token:</strong> <code class="cw-code">render_verified_phase_result</code>
<strong class="cw-strong">Output count:</strong> <code class="cw-code">1</code></p><p class="cw-p"><strong class="cw-strong">Output:</strong> Hiển thị entropy, expected lag, top-one frequency và coverage.</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Ý nghĩa:</strong> Định lượng khoảng quá khứ mà query cuối ưu tiên.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 140 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 55 · NOTEBOOK WALKTHROUGH</span><a id="cell-141"></a>Cell 141–142 — Phase 55 — Historical V1 Head Comparison Results</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 141–142</a> — heading ID <code class="cw-code">644dbb41</code>, output cell 142 ID <code class="cw-code">05ec07a2</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/results_rebuild.py">results_rebuild.py</a> — <code class="cw-code">render_verified_phase_result(55, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/head_comparison/head_comparison_summary.json">head_comparison_summary.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1 HISTORICAL ATTENTION</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Phạm vi khoa học:</strong> <code class="cw-code">V1 HISTORICAL ATTENTION</code>. Nội dung này không phải attention evidence của final V2 vì V2 Test attention tensors không được tạo.
<strong class="cw-strong">Cell 141 (markdown) ID:</strong> <code class="cw-code">644dbb41</code>
<strong class="cw-strong">Cell 142 (code) ID:</strong> <code class="cw-code">05ec07a2</code>
<strong class="cw-strong">Source token:</strong> <code class="cw-code">render_verified_phase_result</code>
<strong class="cw-strong">Output count:</strong> <code class="cw-code">1</code></p><p class="cw-p"><strong class="cw-strong">Output:</strong> So sánh JSD, cosine và Wasserstein giữa attention heads.</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Ý nghĩa:</strong> Xác định head chuyên biệt hoặc dư thừa.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 142 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 56 · NOTEBOOK WALKTHROUGH</span><a id="cell-143"></a>Cell 143–144 — Phase 56 — Historical V1 Error-Conditioned Attention Results</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 143–144</a> — heading ID <code class="cw-code">de13c9d0</code>, output cell 144 ID <code class="cw-code">5e9eec01</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/results_rebuild.py">results_rebuild.py</a> — <code class="cw-code">render_verified_phase_result(56, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/error_conditioned_attention/error_conditioned_attention_summary.json">error_conditioned_attention_summary.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1 HISTORICAL ATTENTION</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Phạm vi khoa học:</strong> <code class="cw-code">V1 HISTORICAL ATTENTION</code>. Nội dung này không phải attention evidence của final V2 vì V2 Test attention tensors không được tạo.
<strong class="cw-strong">Cell 143 (markdown) ID:</strong> <code class="cw-code">de13c9d0</code>
<strong class="cw-strong">Cell 144 (code) ID:</strong> <code class="cw-code">5e9eec01</code>
<strong class="cw-strong">Source token:</strong> <code class="cw-code">render_verified_phase_result</code>
<strong class="cw-strong">Output count:</strong> <code class="cw-code">1</code></p><p class="cw-p"><strong class="cw-strong">Output:</strong> So sánh attention giữa high-error và low-error samples cùng decile trends.</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Ý nghĩa:</strong> Liên kết interpretability với model error mà không suy diễn causal claim.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 144 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 57 · NOTEBOOK WALKTHROUGH</span><a id="cell-145"></a>Cell 145–146 — Phase 57 — Historical V1 Seed-Stability Attention Results</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 145–146</a> — heading ID <code class="cw-code">5627f983</code>, output cell 146 ID <code class="cw-code">4293c1aa</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/results_rebuild.py">results_rebuild.py</a> — <code class="cw-code">render_verified_phase_result(57, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/seed_stability_attention/seed_stability_attention_summary.json">seed_stability_attention_summary.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V1 HISTORICAL ATTENTION</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Phạm vi khoa học:</strong> <code class="cw-code">V1 HISTORICAL ATTENTION</code>. Nội dung này không phải attention evidence của final V2 vì V2 Test attention tensors không được tạo.
<strong class="cw-strong">Cell 145 (markdown) ID:</strong> <code class="cw-code">5627f983</code>
<strong class="cw-strong">Cell 146 (code) ID:</strong> <code class="cw-code">4293c1aa</code>
<strong class="cw-strong">Source token:</strong> <code class="cw-code">render_verified_phase_result</code>
<strong class="cw-strong">Output count:</strong> <code class="cw-code">1</code></p><p class="cw-p"><strong class="cw-strong">Output:</strong> Hiển thị cross-seed attention stability và canonical head mapping.</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Ý nghĩa:</strong> Đánh giá attention có ổn định giữa các seed hay không.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở trực tiếp Cell 146 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 58 · NOTEBOOK WALKTHROUGH</span><a id="cell-147"></a>Cell 147–148 — Phase 58 — Final V2 Results Summary</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 147–148</a> — heading ID <code class="cw-code">332dd9c8</code>, output cell 148 ID <code class="cw-code">9f864b91</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/results_rebuild.py">results_rebuild.py</a> — <code class="cw-code">render_verified_phase_result(58, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/model_improvement_v2/final_reporting_analysis/phase58_final_summary.json">phase58_final_summary.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V2 FINAL</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Cell 147 (markdown) ID:</strong> <code class="cw-code">332dd9c8</code>
<strong class="cw-strong">Cell 148 (code) ID:</strong> <code class="cw-code">9f864b91</code>
<strong class="cw-strong">Renderer:</strong> <code class="cw-code">render_verified_phase_result(58, PROJECT_ROOT)</code>
<strong class="cw-strong">Output count kỳ vọng:</strong> <code class="cw-code">1</code></p><p class="cw-p"><strong class="cw-strong">Output:</strong> V2 final summary rút gọn từ final lock, Step 17 benchmark, MAPE addendum và Phase 48–51 reporting artifacts.</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Ý nghĩa:</strong> Tập hợp kết quả final V2; Phase 52–57 chỉ được nhắc như <code class="cw-code">Historical V1 interpretability evidence</code>.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở Phase 58 trong notebook</a></p><hr class="cw-rule"/></article><article class="cw-card cw-cell-card"><div class="cw-card-rail"><span class="cw-card-dot"></span></div><h3 class="cw-heading cw-h3"><span class="cw-card-kicker">PHASE 59 · NOTEBOOK WALKTHROUGH</span><a id="cell-149"></a>Cell 149–150 — Phase 59 — Final V2 Conclusions</h3><p class="cw-p"><strong class="cw-strong">Liên kết hiện tại:</strong></p><ul class="cw-list">
<li class="cw-li">Notebook: <a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Cells 149–150</a> — heading ID <code class="cw-code">251da856</code>, output cell 150 ID <code class="cw-code">76de7c69</code>.</li>
<li class="cw-li">Python: <a class="cw-link" href="../../src/course_work/reporting/results_rebuild.py">results_rebuild.py</a> — <code class="cw-code">render_verified_phase_result(59, PROJECT_ROOT)</code>.</li>
<li class="cw-li">Artifact: <a class="cw-link" href="../../artifacts/model_improvement_v2/final_reporting_analysis/phase59_final_conclusions.json">phase59_final_conclusions.json</a>.</li>
<li class="cw-li">Lineage: <code class="cw-code">V2 FINAL</code>.</li>
</ul><p class="cw-p"><strong class="cw-strong">Cell 149 (markdown) ID:</strong> <code class="cw-code">251da856</code>
<strong class="cw-strong">Cell 150 (code) ID:</strong> <code class="cw-code">76de7c69</code>
<strong class="cw-strong">Renderer:</strong> <code class="cw-code">render_verified_phase_result(59, PROJECT_ROOT)</code>
<strong class="cw-strong">Output count kỳ vọng:</strong> <code class="cw-code">1</code></p><p class="cw-p"><strong class="cw-strong">Output:</strong> Kết luận cuối cho <code class="cw-code">V2 Equal-weight Ensemble = FINAL_LOCKED_POLICY</code>, limitations và reporting status.</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Ý nghĩa:</strong></p><ul class="cw-list">
<li class="cw-li">Không có post-Test retuning và không chọn best seed.</li>
<li class="cw-li">Không đưa ra claim về V2 attention từ Phase 52–57; artifact ghi <code class="cw-code">attention_conclusion_for_v2=NOT_CLAIMED</code>.</li>
</ul><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở Phase 59 trong notebook</a></p><hr class="cw-rule"/><hr class="cw-rule"/></article></section><section class="cw-section cw-section-v2" data-section="9"><div class="cw-section-heading"><span class="cw-section-index">09</span><h2 class="cw-heading cw-h2"><a id="model-improvement-v2"></a>MODEL_IMPROVEMENT_V2 (Cells 151–152)</h2></div><p class="cw-p"><strong class="cw-strong">Cell 151 (markdown) ID:</strong> <code class="cw-code">v2closure</code>
<strong class="cw-strong">Cell 152 (code) ID:</strong> <code class="cw-code">v2closedash</code></p><p class="cw-p"><strong class="cw-strong">Python:</strong> <a class="cw-link" href="../../src/course_work/reporting/results_rebuild.py">results_rebuild.py</a> — <code class="cw-code">render_verified_v2_results(PROJECT_ROOT)</code>.</p><p class="cw-p"><strong class="cw-strong">Artifacts chính:</strong></p><ul class="cw-list">
<li class="cw-li"><a class="cw-link" href="../../artifacts/model_improvement_v2/final_model_lock/v2_final_model_lock.json">v2_final_model_lock.json</a></li>
<li class="cw-li"><a class="cw-link" href="../../artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics_with_mape.json">step17_metrics_with_mape.json</a></li>
<li class="cw-li"><a class="cw-link" href="../../artifacts/model_improvement_v2/model_improvement_v2_final_closure.json">model_improvement_v2_final_closure.json</a></li>
</ul><p class="cw-p"><strong class="cw-strong">Output:</strong> Tổng quan E01–E20, Step 14A/14B, Step 16 final lock/refit, Step 17 <code class="cw-code">POST_HOC_V2_BENCHMARK</code> và final closure.</p><p class="cw-p cw-semantic-note"><strong class="cw-strong">Ý nghĩa:</strong> Final policy là <code class="cw-code">V2 Equal-weight Ensemble = FINAL_LOCKED_POLICY</code>, dùng seeds <code class="cw-code">42, 123, 2026</code> với weights <code class="cw-code">[1/3,1/3,1/3]</code>. Không chọn best seed và không post-Test retuning.</p><p class="cw-p"><a class="cw-link" href="../../notebook_course_work/CourseWork.ipynb">Mở Cells 151–152 trong notebook</a></p></section><section class="cw-section" data-section="10"><div class="cw-section-heading"><span class="cw-section-index">10</span><h2 class="cw-heading cw-h2">Tài Liệu Liên Quan</h2></div><div class="cw-table-wrap"><table class="cw-table">
<thead>
<tr>
<th>File</th>
<th>Mô tả</th>
</tr>
</thead>
<tbody>
<tr>
<td><a class="cw-link" href="../current_flow/CURRENT_FLOW_SUMMARY.md"><code class="cw-code">CURRENT_FLOW_SUMMARY.md</code></a></td>
<td>Tổng quan kiến trúc</td>
</tr>
<tr>
<td><a class="cw-link" href="../current_flow/PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md"><code class="cw-code">PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md</code></a></td>
<td>Chi tiết phases 1-33</td>
</tr>
<tr>
<td><a class="cw-link" href="../current_flow/PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md"><code class="cw-code">PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md</code></a></td>
<td>Chi tiết phases 34-59</td>
</tr>
</tbody>
</table></div><hr class="cw-rule"/></section><section class="cw-section" data-section="11"><div class="cw-section-heading"><span class="cw-section-index">11</span><h2 class="cw-heading cw-h2">Cách Sử Dụng File Này</h2></div><ol class="cw-list cw-ordered">
<li class="cw-li"><strong class="cw-strong">Khi đọc notebook:</strong> Mở file này song song, mỗi cell gặp output khó hiểu thì tra cứu cell tương ứng.</li>
<li class="cw-li"><strong class="cw-strong">Khi debug:</strong> Tìm cell ID của phase đang lỗi, đọc phần giải thích để biết output kỳ vọng.</li>
<li class="cw-li"><strong class="cw-strong">Khi present:</strong> Mở notebook qua link relative, sau đó tìm <code class="cw-code">cell.id</code> được ghi trong walkthrough; dùng liên kết <code class="cw-code">Giải thích</code> để quay lại section tương ứng.</li>
</ol><p class="cw-p"><strong class="cw-strong">Phiên bản:</strong> 16/09/2026 — baseline chi tiết được giữ nguyên và đồng bộ metadata với notebook/artifacts hiện tại.</p></section></div>
<footer class="cw-footer" id="cw-end">
<div class="cw-footer-content">
<div class="cw-footer-kicker">END OF WALKTHROUGH · 153 CELLS</div>
<h2>From data integrity → modeling → sweeps → final V2 evidence.</h2>
<p>Notebook navigation, cell IDs, artifact links và benchmark lineage được giữ lại để tra cứu chính xác.</p>
<a class="cw-backtop" href="#cw-top">↑ Back to top</a>
</div>
<div aria-hidden="true" class="cw-floating-icons">
<span style="--i:0">🐍</span><span style="--i:1">🔥</span><span style="--i:2">⚡</span><span style="--i:3">📈</span><span style="--i:4">✓</span>
</div>
<svg aria-hidden="true" class="cw-wave" preserveaspectratio="none" viewbox="0 0 1440 240">
<path class="cw-wave-blue" d="M0,90 C240,170 420,15 700,92 C980,170 1170,35 1440,105 L1440,240 L0,240 Z" fill="#2c90e8" opacity="0.70"></path>
<path class="cw-wave-yellow" d="M0,150 C270,85 510,205 770,135 C1020,70 1220,170 1440,125 L1440,240 L0,240 Z" fill="#f7c948" opacity="0.92"></path>
<path class="cw-wave-white" d="M0,185 C300,120 560,220 850,168 C1110,120 1270,205 1440,165 L1440,240 L0,240 Z" fill="#ffffff" opacity="0.96"></path>
</svg>
</footer>
</div>