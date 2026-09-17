<!-- Refactored visual Markdown. CSS/SVG only; no JavaScript. Source content, links and scientific lineage preserved. -->
<style>
:root{--rd-ink:#101828;--rd-muted:#475467;--rd-blue:#0b63ce;--rd-blue2:#2f80ed;--rd-cyan:#4cc9f0;--rd-yellow:#f5c84b;--rd-yellow-soft:#fff6c8;--rd-paper:#fff;--rd-soft:#f6faff;--rd-line:#d8e7f5;--rd-green:#10b981;--rd-green-soft:#eafbf4;--rd-purple:#7c3aed;--rd-red:#d92d20}
*{box-sizing:border-box}html{scroll-behavior:smooth}.rd-shell{max-width:1260px;margin:0 auto;padding:24px 20px 0;background:linear-gradient(180deg,#fff 0%,#f7fbff 45%,#fffdf4 100%);color:var(--rd-ink);font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;line-height:1.72}
.rd-hero{position:relative;isolation:isolate;overflow:hidden;border:1px solid #cfe2f7;border-radius:30px;padding:42px 38px 34px;background:radial-gradient(circle at 86% 11%,rgba(245,200,75,.34),transparent 24%),radial-gradient(circle at 8% 7%,rgba(76,201,240,.24),transparent 29%),linear-gradient(135deg,#fff 0%,#f3f9ff 58%,#fff9dc 100%);box-shadow:0 22px 65px rgba(15,81,145,.12)}.rd-hero:before,.rd-hero:after{content:"";position:absolute;border-radius:50%;z-index:-1}.rd-hero:before{width:250px;height:250px;right:-84px;bottom:-116px;border:1px solid rgba(11,99,206,.18);animation:rdPulse 5s ease-in-out infinite}.rd-hero:after{width:118px;height:118px;right:-16px;bottom:-38px;border:1px solid rgba(245,200,75,.62);animation:rdPulse 4s ease-in-out infinite reverse}.rd-kicker{font-size:12px;font-weight:900;letter-spacing:.18em;text-transform:uppercase;color:#0758b3}.rd-title{margin:7px 0 10px!important;color:#0b1220!important;font-size:clamp(34px,5vw,58px)!important;line-height:1.04!important;border:0!important}.rd-subtitle{max-width:950px;margin:0;color:#344054;font-size:16px}.rd-subtitle strong{color:#0b63ce}
.rd-badges{display:flex;flex-wrap:wrap;gap:10px;margin-top:22px}.rd-badge{display:inline-flex;align-items:center;gap:8px;padding:8px 12px;border:1px solid #d8e7f6;border-radius:999px;background:rgba(255,255,255,.9);font-size:13px;font-weight:750;box-shadow:0 6px 18px rgba(15,81,145,.08);transition:transform .22s ease,box-shadow .22s ease}.rd-badge:hover{transform:translateY(-3px);box-shadow:0 10px 24px rgba(15,81,145,.14)}.rd-badge svg{width:19px;height:19px;display:block}
.rd-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:22px}.rd-stat{padding:14px 15px;border-radius:16px;background:rgba(255,255,255,.84);border:1px solid #dce9f6;backdrop-filter:blur(4px)}.rd-stat-value{font-size:22px;font-weight:900;line-height:1.05;color:#0b63ce}.rd-stat:nth-child(3) .rd-stat-value{color:#7c3aed}.rd-stat:nth-child(4) .rd-stat-value{color:#087a57}.rd-stat-label{margin-top:6px;font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#667085}
.rd-result-strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:14px}.rd-result{padding:12px 14px;border:1px solid #cce9dd;border-radius:14px;background:linear-gradient(135deg,#f5fffb,#fffdf1)}.rd-result strong{display:block;font-size:19px;color:#087a57}.rd-result span{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#667085;font-weight:800}
.rd-hero-flow{display:grid;grid-template-columns:1fr auto 1fr auto 1fr auto 1fr auto 1fr;gap:8px;align-items:center;margin-top:22px}.rd-flow-node{padding:12px 10px;border-radius:14px;border:1px solid #d8e6f4;background:#fff;font-weight:800;text-align:center;box-shadow:0 5px 16px rgba(15,81,145,.06)}.rd-flow-node small{display:block;margin-top:2px;color:#667085;font-weight:650}.rd-flow-arrow{font-size:20px;color:#0b63ce;animation:rdArrow 1.8s ease-in-out infinite}
.rd-document{padding:10px 0 40px}.rd-section{position:relative;margin:20px 0;padding:25px 27px;background:rgba(255,255,255,.97);border:1px solid var(--rd-line);border-radius:21px;box-shadow:0 8px 28px rgba(15,81,145,.065);transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease}.rd-section:hover{transform:translateY(-2px);box-shadow:0 15px 36px rgba(15,81,145,.105);border-color:#bdd9f3}.rd-heading{scroll-margin-top:20px;color:#101828}.rd-h1{display:none}.rd-h2{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:0 0 15px!important;padding:0 0 12px;border-bottom:3px solid transparent;border-image:linear-gradient(90deg,var(--rd-blue),var(--rd-cyan),var(--rd-yellow),transparent) 1;font-size:27px!important}.rd-h3{margin-top:24px!important;color:#163a63!important;font-size:20px!important}.rd-h4{color:#36526f!important}.rd-section-pill{display:inline-flex;align-items:center;padding:4px 8px;border-radius:999px;background:#eaf4ff;border:1px solid #c9def4;color:#0758b3;font-size:10px;font-weight:900;letter-spacing:.08em;text-transform:uppercase;white-space:nowrap}
.rd-toc-card{background:linear-gradient(135deg,#fbfdff,#f3f9ff 68%,#fffaf0)}.rd-toc-card:after{content:"Repository Map";position:absolute;top:18px;right:22px;padding:5px 9px;border-radius:999px;background:#fff3b8;color:#7c5700;font-size:10px;font-weight:900;letter-spacing:.08em;text-transform:uppercase}.rd-toc-list{columns:2;column-gap:36px}.rd-toc-list li{break-inside:avoid;margin:7px 0}.rd-results-card{background:radial-gradient(circle at 94% 7%,rgba(16,185,129,.12),transparent 27%),linear-gradient(135deg,#fff,#f5fffb 65%,#fffbea)}.rd-status-card{background:radial-gradient(circle at 94% 7%,rgba(76,201,240,.16),transparent 28%),#fff}.rd-arch-card,.rd-layers-card,.rd-flow-card{background:linear-gradient(135deg,#fff,#f7fbff)}.rd-phases-card{background:radial-gradient(circle at 97% 4%,rgba(124,58,237,.08),transparent 24%),#fff}.rd-data-card,.rd-protocol-card{background:linear-gradient(135deg,#fff,#f9fcff 68%,#fffdf2)}.rd-complete-card{background:linear-gradient(135deg,#fff,#f3fff9)}.rd-limit-card{background:linear-gradient(135deg,#fffdf5,#fff)}
.rd-current-note{margin:20px 0!important;padding:18px 19px!important;border:1px solid #bdebd9!important;border-left:5px solid #10b981!important;background:linear-gradient(90deg,#ecfbf5,#fffdf1)!important;border-radius:0 15px 15px 0;color:#26364a}.rd-current-label{display:inline-flex;align-items:center;gap:8px;margin-bottom:7px;padding:4px 9px;border-radius:999px;background:#dff8ed;color:#087a57;font-size:10px;font-weight:900;letter-spacing:.09em}.rd-current-label span{display:grid;place-items:center;width:18px;height:18px;border-radius:50%;background:#10b981;color:#fff}.rd-note{margin:18px 0!important;padding:16px 18px!important;border-left:4px solid var(--rd-blue)!important;background:linear-gradient(90deg,#edf6ff,#fffdf2)!important;border-radius:0 13px 13px 0;color:#26364a}
.rd-nav-notice{display:grid;grid-template-columns:auto 1fr;gap:14px;align-items:start;margin:18px 0;padding:17px 18px;border-radius:16px;border:1px solid #cfe2f7;background:linear-gradient(90deg,#edf6ff,#fffdf0);box-shadow:0 7px 20px rgba(15,81,145,.06)}.rd-nav-icon{width:38px;height:38px;display:grid;place-items:center;border-radius:11px;background:#0b63ce;color:white;font-size:20px;font-weight:900}.rd-nav-notice strong{color:#0758b3}.rd-nav-notice p{margin:3px 0 0;color:#475467}.rd-notebook-link{display:inline-flex!important;align-items:center;padding:2px 8px;border-radius:999px;background:#eaf4ff;border:1px solid #cfe2f7;text-decoration:none!important}.rd-notebook-link:hover{background:#ddecff;text-decoration:none!important}
.rd-policy-ribbon{display:grid;grid-template-columns:auto 1fr auto;gap:12px;align-items:center;margin:16px 0 20px;padding:15px 16px;border:1px solid #bcead9;border-radius:16px;background:linear-gradient(90deg,#ebfbf5,#f8fffc 60%,#fff8d8);box-shadow:0 8px 22px rgba(16,185,129,.08)}.rd-policy-icon{width:37px;height:37px;display:grid;place-items:center;border-radius:50%;background:#10b981;color:#fff;font-weight:900}.rd-policy-ribbon strong{display:block;color:#087a57;font-size:14px;letter-spacing:.04em}.rd-policy-ribbon span:not(.rd-policy-icon):not(.rd-policy-lock){display:block;color:#475467;font-size:13px}.rd-policy-lock{padding:5px 9px;border-radius:999px;background:#eafbf4;border:1px solid #bcead9;color:#087a57;font-size:10px;font-weight:900;letter-spacing:.08em}
.rd-rule{border:0!important;height:1px!important;background:linear-gradient(90deg,transparent,#93c5fd,#facc15,transparent)!important;margin:28px 0!important}.rd-link{color:#075fbe!important;text-decoration:none!important;font-weight:650}.rd-link:hover{text-decoration:underline!important;text-decoration-thickness:1.5px!important}.rd-inline-code{padding:.14em .42em;border:1px solid #d9e7f4;border-radius:6px;background:#f3f8fc;color:#854d0e;font-size:.92em}.rd-code-blue{background:#eaf4ff!important;color:#0758b3!important;border-color:#c9def4!important}.rd-code-yellow{background:#fff6c9!important;color:#7b5600!important;border-color:#f0dc86!important}.rd-code-green{background:#eafbf4!important;color:#087a57!important;border-color:#bdebd9!important}.rd-code{overflow:auto;padding:17px 18px;border:1px solid #cfe0ef;border-radius:14px;background:#f7fbff;box-shadow:inset 4px 0 0 #4cc9f0;white-space:pre}.rd-code code{background:transparent!important;color:#111827!important}.rd-mermaid{background:linear-gradient(135deg,#f7fbff,#fffdf5);box-shadow:inset 4px 0 0 #0b63ce}
.rd-table-wrap{overflow-x:auto;margin:18px 0;border:1px solid #d7e6f4;border-radius:14px;box-shadow:0 5px 18px rgba(15,81,145,.06)}.rd-table{width:100%;border-collapse:collapse;background:#fff;font-size:14px}.rd-table th{padding:11px 12px;background:linear-gradient(90deg,#eaf5ff,#fff6c9);color:#172033;text-align:left;border-bottom:1px solid #cbddeb}.rd-table td{padding:10px 12px;border-bottom:1px solid #edf2f7;vertical-align:top}.rd-table tr:last-child td{border-bottom:0}.rd-table tbody tr:hover{background:#f8fcff}ul,ol{padding-left:24px}li::marker{color:#0b63ce}strong{color:#111827}
.rd-footer{position:relative;overflow:hidden;margin:24px -20px 0;padding:66px 24px 94px;text-align:center;background:linear-gradient(180deg,#f8fcff,#e9f6ff 54%,#fff4bf);border-top:1px solid #cfe4f7}.rd-footer h3{margin:0;color:#101828;font-size:25px}.rd-footer p{margin:7px auto 0;max-width:820px;color:#475467}.rd-floaters{display:flex;justify-content:center;gap:18px;margin:23px 0 4px}.rd-floater{font-size:29px;display:inline-block;animation:rdFloat 3s ease-in-out infinite}.rd-floater:nth-child(2){animation-delay:.3s}.rd-floater:nth-child(3){animation-delay:.6s}.rd-floater:nth-child(4){animation-delay:.9s}.rd-floater:nth-child(5){animation-delay:1.2s}.rd-wave{position:absolute;left:-1%;right:-1%;bottom:-3px;width:102%;height:78px;opacity:.55}.rd-wave path:first-child{animation:rdWave 5s ease-in-out infinite alternate}.rd-wave path:last-child{animation:rdWave 6s ease-in-out infinite alternate-reverse}
@keyframes rdPulse{0%,100%{transform:scale(1);opacity:.62}50%{transform:scale(1.12);opacity:1}}@keyframes rdFloat{0%,100%{transform:translateY(0) rotate(0)}50%{transform:translateY(-11px) rotate(4deg)}}@keyframes rdArrow{0%,100%{transform:translateX(0);opacity:.65}50%{transform:translateX(4px);opacity:1}}@keyframes rdWave{from{transform:translateX(-8px)}to{transform:translateX(8px)}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important;scroll-behavior:auto!important}}@media(max-width:860px){.rd-stats,.rd-result-strip{grid-template-columns:repeat(2,minmax(0,1fr))}.rd-hero-flow{grid-template-columns:1fr}.rd-flow-arrow{transform:rotate(90deg)}.rd-toc-list{columns:1}.rd-policy-ribbon{grid-template-columns:auto 1fr}.rd-policy-lock{grid-column:2}.rd-section{padding:20px 18px}}@media(max-width:620px){.rd-shell{padding:12px 10px 0}.rd-hero{padding:26px 20px;border-radius:20px}.rd-stats,.rd-result-strip{grid-template-columns:1fr 1fr}.rd-h2{font-size:23px!important}.rd-footer{margin-left:-10px;margin-right:-10px}}
</style>
<div class="rd-shell">
<div class="rd-hero">
<div class="rd-kicker">Deep Learning · Time-Series Regression · Reproducible Coursework</div>
<h1 class="rd-title">Deep Learning Lab & Practice</h1>
<p class="rd-subtitle">Repository landing page cho <strong>COURSE_WORK</strong>: từ data integrity, sequence modeling và controlled sweeps đến <strong>MODEL_IMPROVEMENT_V2</strong>, immutable final lock và reporting-only analysis.</p>
<div class="rd-badges"><span class="rd-badge"><svg viewBox="0 0 24 24" aria-hidden="true"><path fill="#3776AB" d="M12 2c-4 0-4 2-4 4v2h8v1H6c-2 0-4 1-4 4s2 4 4 4h2v-3c0-2 2-4 4-4h6c2 0 4-2 4-4s-2-4-4-4z"/><circle cx="10" cy="5" r="1" fill="#fff"/><path fill="#FFD43B" d="M12 22c4 0 4-2 4-4v-2H8v-1h10c2 0 4-1 4-4s-2-4-4-4h-2v3c0 2-2 4-4 4H6c-2 0-4 2-4 4s2 4 4 4z"/><circle cx="14" cy="19" r="1" fill="#fff"/></svg>Python 3.10</span><span class="rd-badge">🔥 PyTorch</span><span class="rd-badge">◫ Jupyter</span><span class="rd-badge">⚡ Transformer</span><span class="rd-badge">⌁ Time Series</span><span class="rd-badge">◈ Artifacts</span><span class="rd-badge">✓ Reproducible</span></div>
<div class="rd-stats"><div class="rd-stat"><div class="rd-stat-value">60</div><div class="rd-stat-label">Phases 0–59</div></div><div class="rd-stat"><div class="rd-stat-value">153</div><div class="rd-stat-label">Notebook cells</div></div><div class="rd-stat"><div class="rd-stat-value">19</div><div class="rd-stat-label">Controlled sweeps</div></div><div class="rd-stat"><div class="rd-stat-value">V2 FINAL</div><div class="rd-stat-label">Locked policy</div></div></div>
<div class="rd-result-strip"><div class="rd-result"><strong>61.608937</strong><span>RMSE · Wh</span></div><div class="rd-result"><strong>25.897130</strong><span>MAE · Wh</span></div><div class="rd-result"><strong>21.519219%</strong><span>MAPE</span></div><div class="rd-result"><strong>0.540364</strong><span>R²</span></div></div>
<div class="rd-hero-flow"><div class="rd-flow-node">Data<small>Audit · Split</small></div><div class="rd-flow-arrow">→</div><div class="rd-flow-node">Models<small>LSTM · Transformer</small></div><div class="rd-flow-arrow">→</div><div class="rd-flow-node">Sweeps<small>S1–S19</small></div><div class="rd-flow-arrow">→</div><div class="rd-flow-node">V2 Improvement<small>E01–E20</small></div><div class="rd-flow-arrow">→</div><div class="rd-flow-node">Final Policy<small>3-seed ensemble</small></div></div>
</div>
<div class="rd-document"><h1 class="rd-heading rd-h1">Deep Learning Lab and Practice</h1>
<blockquote class="rd-current-note"><div class="rd-current-label"><span>✓</span> CURRENT VERIFIED STATUS</div>
<p><strong>Current verified status:</strong> final policy là <code class="rd-inline-code rd-code-green">V2 Equal-weight Ensemble = FINAL_LOCKED_POLICY</code>, tương ứng <code class="rd-inline-code">mean(E14-M1 final-refit seeds 42, 123, 2026)</code> với weights <code class="rd-inline-code">[1/3,1/3,1/3]</code>. Step 17 là <code class="rd-inline-code rd-code-yellow">POST_HOC_V2_BENCHMARK</code>, không phải unseen unbiased Test. V1 được giữ làm historical baseline/provenance.</p>
</blockquote><div class="rd-nav-notice"><div class="rd-nav-icon">↗</div><div><strong>Notebook navigation</strong><p>Links to <code>CourseWork.ipynb</code> open the notebook file only. Markdown/VS Code/Jupyter links do not reliably jump to an exact notebook cell; use the notebook walkthrough for exact cell mapping.</p></div></div>
<p>Repository này lưu trữ bài tập thực hành Deep Learning và coursework chính về dự báo mức tiêu thụ năng lượng bằng mô hình chuỗi thời gian đa biến. Phần được tài liệu hóa đầy đủ trong README này là <code class="rd-inline-code">COURSE_WORK</code>, nơi triển khai toàn bộ quy trình từ kiểm tra dữ liệu, xây dựng đặc trưng, huấn luyện mô hình, tối ưu siêu tham số, đánh giá trên tập Test khóa, đến phân tích sai số và attention.</p>
<section class="rd-section rd-toc-card"><h2 class="rd-heading rd-h2" id="mục-lục"><span class="rd-section-pill">NAVIGATION</span>Mục lục</h2>
<ul class="rd-toc-list">
<li><a class="rd-link" href="#tổng-quan-coursework">Tổng quan coursework</a></li>
<li><a class="rd-link" href="#bài-toán-và-mục-tiêu">Bài toán và mục tiêu</a></li>
<li><a class="rd-link" href="#dữ-liệu">Dữ liệu</a></li>
<li><a class="rd-link" href="#giao-thức-thực-nghiệm">Giao thức thực nghiệm</a></li>
<li><a class="rd-link" href="#mô-hình-và-chỉ-số-đánh-giá">Mô hình và chỉ số đánh giá</a></li>
<li><a class="rd-link" href="#kết-quả-đã-khóa">Kết quả đã khóa</a></li>
<li><a class="rd-link" href="#kiến-trúc-thư-mục">Kiến trúc thư mục</a></li>
<li><a class="rd-link" href="#trách-nhiệm-của-từng-tầng">Trách nhiệm của từng tầng</a></li>
<li><a class="rd-link" href="#luồng-xử-lý-đầu-cuối">Luồng xử lý đầu cuối</a></li>
<li><a class="rd-link" href="#chi-tiết-phase-059">Chi tiết Phase 0–59</a></li>
<li><a class="rd-link" href="#cơ-chế-artifact-log-và-selective-resume">Cơ chế artifact, log và selective resume</a></li>
<li><a class="rd-link" href="#vai-trò-của-notebook">Vai trò của notebook</a></li>
<li><a class="rd-link" href="#cài-đặt-và-sử-dụng">Cài đặt và sử dụng</a></li>
<li><a class="rd-link" href="#kiểm-thử-và-tái-lập">Kiểm thử và tái lập</a></li>
<li><a class="rd-link" href="#tài-liệu-dự-án">Tài liệu dự án</a></li>
<li><a class="rd-link" href="#những-gì-dự-án-đã-hoàn-thành">Những gì dự án đã hoàn thành</a></li>
<li><a class="rd-link" href="#giới-hạn-và-hướng-phát-triển">Giới hạn và hướng phát triển</a></li>
</ul>
</section><section class="rd-section rd-overview-card"><h2 class="rd-heading rd-h2" id="tổng-quan-coursework"><span class="rd-section-pill">OVERVIEW</span>Tổng quan coursework</h2>
<p><code class="rd-inline-code">COURSE_WORK</code> giải quyết bài toán hồi quy chuỗi thời gian đa biến trên bộ dữ liệu UCI Appliances Energy Prediction. Hệ thống sử dụng các quan sát lịch sử về nhiệt độ, độ ẩm, điều kiện thời tiết ngoài trời, ánh sáng và các biến liên quan để dự đoán điện năng tiêu thụ của thiết bị gia dụng ở bước thời gian kế tiếp.</p>
<p>Các thành phần chính của coursework gồm:</p>
<ul>
<li>Persistence baseline để thiết lập mức tham chiếu tối thiểu.</li>
<li>LSTM baseline để đại diện cho mô hình tuần tự hồi quy.</li>
<li>Transformer Encoder cho hồi quy sequence-to-one.</li>
<li>Chuỗi sweep có kiểm soát để lựa chọn đặc trưng và siêu tham số.</li>
<li>Giao thức chia dữ liệu theo thời gian và chuẩn hóa chỉ học trên Train.</li>
<li>Cơ chế khóa Test cho đến giai đoạn đánh giá cuối cùng.</li>
<li>Đánh giá bằng MAE, RMSE, R² và MAPE bổ sung.</li>
<li>Phân tích dự đoán, residual, chế độ tiêu thụ, trường hợp sai số lớn và attention.</li>
<li>Artifact, checksum, signoff và processing log cho khả năng tái lập và chạy tiếp theo trạng thái đã lưu.</li>
</ul>
<p>Notebook trung tâm của dự án là <a class="rd-link rd-notebook-link" href="COURSE_WORK/notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Markdown links do not reliably deep-link to a specific notebook cell.">CourseWork.ipynb</a>. Notebook đóng vai trò điều phối và trực quan hóa; phần xử lý nghiệp vụ được đặt trong package <code class="rd-inline-code">src/course_work</code>.</p>
</section><section class="rd-section rd-problem-card"><h2 class="rd-heading rd-h2" id="bài-toán-và-mục-tiêu"><span class="rd-section-pill">PROBLEM</span>Bài toán và mục tiêu</h2>
<h3 class="rd-heading rd-h3" id="định-nghĩa-bài-toán">Định nghĩa bài toán</h3>
<p>Đây là bài toán supervised multivariate time-series regression với cấu trúc sequence-to-one:</p>
<p>[
X[t-L+1:t] \longrightarrow \widehat{y}[t+1]
]</p>
<p>Trong đó:</p>
<ul>
<li><code class="rd-inline-code">X[t-L+1:t]</code> là cửa sổ lịch sử gồm nhiều biến đầu vào.</li>
<li><code class="rd-inline-code">L</code> là độ dài lookback.</li>
<li><code class="rd-inline-code">y</code> là biến mục tiêu <code class="rd-inline-code">Appliances</code>.</li>
<li>Mỗi bước dữ liệu tương ứng 10 phút.</li>
<li>Forecast horizon là 1 bước, tương đương dự báo trước 10 phút.</li>
<li>Đơn vị mục tiêu và các sai số MAE, RMSE là watt-hour (<code class="rd-inline-code">Wh</code>).</li>
</ul>
<h3 class="rd-heading rd-h3" id="mục-tiêu-nghiên-cứu">Mục tiêu nghiên cứu</h3>
<p>Dự án trả lời các câu hỏi chính sau:</p>
<ol>
<li>Transformer Encoder có cải thiện dự báo so với persistence baseline và LSTM baseline hay không.</li>
<li>Nhóm đặc trưng, time feature, target scaling và lookback nào phù hợp nhất với bài toán.</li>
<li>Các lựa chọn pooling, activation, batch size, learning rate, weight decay, dropout, kích thước mô hình, số head, số layer, FFN, loss, epoch cap, gradient clipping và RevIN ảnh hưởng thế nào đến Validation RMSE.</li>
<li>Cấu hình tốt nhất có ổn định qua nhiều seed và qua rolling-origin hay không.</li>
<li>Mô hình hoạt động thế nào trên tập Test bị khóa.</li>
<li>Sai số tập trung ở chế độ tiêu thụ nào và ở các giai đoạn biến động nào.</li>
<li>Attention tập trung vào những vị trí lịch sử nào, khác nhau ra sao giữa các head và có ổn định qua seed hay không.</li>
</ol>
<h3 class="rd-heading rd-h3" id="phạm-vi">Phạm vi</h3>
<p>Coursework tập trung vào một bộ dữ liệu, một hộ gia đình, bài toán dự báo một bước và một quy trình thực nghiệm ngoại tuyến. Dự án không tuyên bố khả năng triển khai production, khả năng tổng quát hóa sang hộ gia đình khác hoặc quan hệ nhân quả từ attention.</p>
</section><section class="rd-section rd-data-card"><h2 class="rd-heading rd-h2" id="dữ-liệu"><span class="rd-section-pill">DATA</span>Dữ liệu</h2>
<h3 class="rd-heading rd-h3" id="nguồn-dữ-liệu">Nguồn dữ liệu</h3>
<ul>
<li>Dataset: <a class="rd-link" href="https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction">UCI Appliances Energy Prediction</a>.</li>
<li>Tệp dữ liệu chính: <code class="rd-inline-code">energydata_complete.csv</code>.</li>
<li>Tần suất lấy mẫu: 10 phút.</li>
<li>Biến mục tiêu: <code class="rd-inline-code">Appliances</code>.</li>
<li>Dạng dữ liệu: chuỗi thời gian đa biến với cảm biến trong nhà, thời tiết ngoài trời, ánh sáng và biến ngẫu nhiên đi kèm bộ dữ liệu gốc.</li>
</ul>
<h3 class="rd-heading rd-h3" id="nguyên-tắc-xử-lý-dữ-liệu">Nguyên tắc xử lý dữ liệu</h3>
<ul>
<li>Giữ nguyên thứ tự thời gian trong mọi bước xử lý.</li>
<li>Kiểm tra schema, kiểu dữ liệu, timestamp, khoảng lấy mẫu, trùng lặp và giá trị thiếu trước khi tạo đặc trưng.</li>
<li>Không dùng thông tin tương lai để tạo đặc trưng cho một thời điểm hiện tại.</li>
<li>Chia Train, Validation và Test theo thời gian, không shuffle trước khi chia.</li>
<li>Fit scaler trên Train rồi áp dụng cùng tham số cho Validation và Test.</li>
<li>Chỉ dùng Validation để lựa chọn mô hình và siêu tham số.</li>
<li>Giữ Test khóa trong V1 cho đến historical Phase 47; V2 chỉ mở old Test tại Step 17 sau khi Step 16 final lock hoàn tất.</li>
</ul>
<h3 class="rd-heading rd-h3" id="phân-bổ-dữ-liệu">Phân bổ dữ liệu</h3>
<p>Giao thức chính sử dụng tỷ lệ thời gian:</p>
<div class="rd-table-wrap"><table class="rd-table">
<thead>
<tr>
<th>Tập dữ liệu</th>
<th style="text-align:right">Tỷ lệ</th>
<th>Vai trò</th>
</tr>
</thead>
<tbody>
<tr>
<td>Train</td>
<td style="text-align:right">70%</td>
<td>Học tham số mô hình và fit các phép biến đổi</td>
</tr>
<tr>
<td>Validation</td>
<td style="text-align:right">15%</td>
<td>Chọn cấu hình, early stopping và so sánh sweep</td>
</tr>
<tr>
<td>Test</td>
<td style="text-align:right">15%</td>
<td>Đánh giá cuối cùng sau khi khóa mô hình</td>
</tr>
</tbody>
</table></div>
<p>Hai giao thức biên cửa sổ được xem xét:</p>
<ul>
<li><code class="rd-inline-code rd-code-blue">WB0_CONTEXT_CARRY_OVER</code>: cho phép cửa sổ của Validation hoặc Test nhận phần lịch sử cần thiết từ đoạn thời gian ngay trước biên, trong khi target vẫn thuộc đúng tập.</li>
<li><code class="rd-inline-code">WB1_STRICT_ISOLATION</code>: chỉ tạo cửa sổ từ dữ liệu nằm hoàn toàn trong từng partition.</li>
</ul>
<p>WB0 là giao thức cuối cùng được khóa cho báo cáo chính.</p>
</section><section class="rd-section rd-protocol-card"><h2 class="rd-heading rd-h2" id="giao-thức-thực-nghiệm"><span class="rd-section-pill">PROTOCOL</span>Giao thức thực nghiệm</h2>
<h3 class="rd-heading rd-h3" id="nguyên-tắc-lựa-chọn">Nguyên tắc lựa chọn</h3>
<ul>
<li>Metric lựa chọn chính là Validation RMSE trên thang đo gốc <code class="rd-inline-code">Wh</code>.</li>
<li>MAE và R² được báo cáo đồng thời để mô tả chất lượng dự báo.</li>
<li>MAPE là metric bổ sung, không thay thế tiêu chí lựa chọn RMSE.</li>
<li>Mỗi sweep chỉ thay đổi yếu tố đang nghiên cứu và giữ cố định cấu hình tham chiếu còn lại.</li>
<li>Nếu có hòa chính xác, quy tắc tie-break của từng phase được áp dụng và lưu trong contract.</li>
<li>Mọi cấu hình thắng được chuyển tiếp qua canonical handoff hoặc reference update.</li>
<li>Test không được dùng để chọn feature set, hyperparameter, seed, epoch, ensemble weight hay checkpoint.</li>
</ul>
<h3 class="rd-heading rd-h3" id="kiểm-soát-leakage">Kiểm soát leakage</h3>
<p>Các lớp bảo vệ chính gồm:</p>
<ul>
<li>Chronological split trước các phép học từ dữ liệu.</li>
<li>Train-only scaling cho cả feature và target.</li>
<li>Cửa sổ dự báo chỉ sử dụng các thời điểm không muộn hơn thời điểm dự báo cho phép.</li>
<li>Validation chịu trách nhiệm lựa chọn; Test chỉ dùng cho đánh giá cuối.</li>
<li>Artifact và checksum ghi lại đầu vào, đầu ra và quan hệ phụ thuộc giữa các phase.</li>
<li>Final model lock đóng băng cấu hình trước khi mở Test.</li>
</ul>
<h3 class="rd-heading rd-h3" id="đơn-vị-thực-nghiệm">Đơn vị thực nghiệm</h3>
<p>Mỗi run có mã định danh, cấu hình, seed, metric, checkpoint và metadata. Experiment registry bảo đảm run không bị ghi đè tùy tiện và có thể truy vết ngược về dữ liệu, scaler, window population và cấu hình đã dùng.</p>
</section><section class="rd-section rd-model-card"><h2 class="rd-heading rd-h2" id="mô-hình-và-chỉ-số-đánh-giá"><span class="rd-section-pill">MODELS &amp; METRICS</span>Mô hình và chỉ số đánh giá</h2>
<h3 class="rd-heading rd-h3" id="persistence-baseline">Persistence baseline</h3>
<p>Persistence dự đoán mức tiêu thụ ở bước kế tiếp bằng giá trị mục tiêu gần nhất:</p>
<p>[
\widehat{y}_{t+1}=y_t
]</p>
<p>Baseline này kiểm tra liệu mô hình học sâu có tạo ra giá trị vượt quá tính tự tương quan ngắn hạn của chuỗi hay không.</p>
<h3 class="rd-heading rd-h3" id="lstm-baseline">LSTM baseline</h3>
<p>LSTM nhận cửa sổ đa biến và tạo dự đoán sequence-to-one. Mô hình đóng vai trò baseline học sâu tuần tự để so sánh với Transformer. Cấu hình LSTM được huấn luyện, tinh chỉnh và kiểm tra robustness trong các phase riêng.</p>
<h3 class="rd-heading rd-h3" id="transformer-encoder">Transformer Encoder</h3>
<p>Transformer sử dụng projection đầu vào, positional encoding, nhiều encoder layer và regression head. Thiết kế hỗ trợ trích xuất attention để phân tích hành vi mô hình sau huấn luyện.</p>
<p>Cấu hình final V2 đã khóa:</p>
<div class="rd-table-wrap"><table class="rd-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Giá trị</th>
</tr>
</thead>
<tbody>
<tr>
<td>Candidate</td>
<td><code class="rd-inline-code">TR_C2_ALT_LOOKBACK_E14_M1</code></td>
</tr>
<tr>
<td>Prediction policy</td>
<td>Equal-weight mean của seeds <code class="rd-inline-code">42, 123, 2026</code></td>
</tr>
<tr>
<td>Ensemble weights</td>
<td><code class="rd-inline-code">[1/3,1/3,1/3]</code></td>
</tr>
<tr>
<td>Feature set</td>
<td><code class="rd-inline-code">FS2_TF1</code></td>
</tr>
<tr>
<td>Số feature</td>
<td>33</td>
</tr>
<tr>
<td>Lookback</td>
<td>72 bước</td>
</tr>
<tr>
<td>Forecast horizon</td>
<td>1 bước</td>
</tr>
<tr>
<td>Boundary protocol</td>
<td><code class="rd-inline-code rd-code-blue">WB0_CONTEXT_CARRY_OVER</code></td>
</tr>
<tr>
<td>Model</td>
<td>Transformer Encoder regression, Post-LN</td>
</tr>
<tr>
<td><code class="rd-inline-code">d_model</code></td>
<td>64</td>
</tr>
<tr>
<td>Attention heads</td>
<td>4</td>
</tr>
<tr>
<td>Encoder layers</td>
<td>2</td>
</tr>
<tr>
<td>FFN width</td>
<td>256</td>
</tr>
<tr>
<td>Dropout</td>
<td>0.10</td>
</tr>
<tr>
<td>Pooling</td>
<td><code class="rd-inline-code">LAST_STEP</code></td>
</tr>
<tr>
<td>Optimizer</td>
<td>AdamW</td>
</tr>
<tr>
<td>Learning rate</td>
<td><code class="rd-inline-code">2e-4</code></td>
</tr>
<tr>
<td>Weight decay</td>
<td><code class="rd-inline-code">1e-3</code></td>
</tr>
<tr>
<td>Loss</td>
<td>Hybrid level-plus-delta, <code class="rd-inline-code">lambda_delta=0.10</code></td>
</tr>
<tr>
<td>Gradient clipping</td>
<td>1.0</td>
</tr>
<tr>
<td>Batch size</td>
<td>16</td>
</tr>
<tr>
<td>Scheduler</td>
<td>OFF</td>
</tr>
<tr>
<td>Final-refit epochs</td>
<td>16 cho cả ba seed</td>
</tr>
<tr>
<td>Final seeds</td>
<td>42, 123, 2026</td>
</tr>
</tbody>
</table></div>
<p>Nguồn cấu hình và checkpoint lineage: <a class="rd-link" href="COURSE_WORK/artifacts/model_improvement_v2/final_model_lock/v2_final_model_lock.json"><code class="rd-inline-code">v2_final_model_lock.json</code></a>.</p>
<h3 class="rd-heading rd-h3" id="chỉ-số-đánh-giá">Chỉ số đánh giá</h3>
<div class="rd-table-wrap"><table class="rd-table">
<thead>
<tr>
<th>Metric</th>
<th>Ý nghĩa</th>
<th>Vai trò</th>
</tr>
</thead>
<tbody>
<tr>
<td>MAE</td>
<td>Sai số tuyệt đối trung bình trên thang <code class="rd-inline-code">Wh</code></td>
<td>Dễ diễn giải theo mức sai lệch trung bình</td>
</tr>
<tr>
<td>RMSE</td>
<td>Căn bậc hai của sai số bình phương trung bình</td>
<td>Metric lựa chọn chính, nhạy với sai số lớn</td>
</tr>
<tr>
<td>R²</td>
<td>Tỷ lệ phương sai mục tiêu được mô hình giải thích</td>
<td>Đánh giá mức cải thiện so với dự đoán trung bình</td>
</tr>
<tr>
<td>MAPE</td>
<td>Sai số phần trăm tuyệt đối trung bình</td>
<td>Metric bổ sung, cần xử lý rõ trường hợp target bằng hoặc gần 0</td>
</tr>
</tbody>
</table></div>
<p>MAPE được bổ sung bằng reporting-only addendum từ frozen Step 17 prediction artifacts. Addendum không train, không chạy model inference mới và không mở lại raw Test source; kết quả nằm tại <a class="rd-link" href="COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics_with_mape.json"><code class="rd-inline-code">step17_metrics_with_mape.json</code></a>.</p>
</section><section class="rd-section rd-results-card"><h2 class="rd-heading rd-h2" id="kết-quả-đã-khóa"><span class="rd-section-pill">V2 FINAL</span>Kết quả đã khóa</h2><div class="rd-policy-ribbon"><span class="rd-policy-icon">✓</span><div><strong>FINAL_LOCKED_POLICY</strong><span>Equal-weight Transformer ensemble · seeds 42 / 123 / 2026 · no post-Test retuning</span></div><span class="rd-policy-lock">LOCKED</span></div>
<h3 class="rd-heading rd-h3" id="final-v2-locked-policy">Final V2 locked policy</h3>
<p>Final policy là <code class="rd-inline-code rd-code-green">V2 Equal-weight Ensemble = FINAL_LOCKED_POLICY</code>. Step 17 đánh giá policy đã khóa trên old Test với nhãn <code class="rd-inline-code rd-code-yellow">POST_HOC_V2_BENCHMARK</code>; kết quả không được diễn giải là unseen unbiased Test và không được dùng để retune.</p>
<div class="rd-table-wrap"><table class="rd-table">
<thead>
<tr>
<th>Model / policy</th>
<th style="text-align:right">RMSE (Wh)</th>
<th style="text-align:right">MAE (Wh)</th>
<th style="text-align:right">MAPE (%)</th>
<th style="text-align:right">R²</th>
<th>Vai trò</th>
</tr>
</thead>
<tbody>
<tr>
<td>V2 Seed 42</td>
<td style="text-align:right">61.262482</td>
<td style="text-align:right">27.292729</td>
<td style="text-align:right">24.187279</td>
<td style="text-align:right">0.545519</td>
<td>Thành phần ensemble</td>
</tr>
<tr>
<td>V2 Seed 123</td>
<td style="text-align:right">63.259959</td>
<td style="text-align:right">27.332280</td>
<td style="text-align:right">23.147923</td>
<td style="text-align:right">0.515398</td>
<td>Thành phần ensemble</td>
</tr>
<tr>
<td>V2 Seed 2026</td>
<td style="text-align:right">62.423633</td>
<td style="text-align:right">25.734231</td>
<td style="text-align:right">20.222825</td>
<td style="text-align:right">0.528127</td>
<td>Thành phần ensemble</td>
</tr>
<tr>
<td><strong>V2 Equal-weight Ensemble</strong></td>
<td style="text-align:right"><strong>61.608937</strong></td>
<td style="text-align:right"><strong>25.897130</strong></td>
<td style="text-align:right"><strong>21.519219</strong></td>
<td style="text-align:right"><strong>0.540364</strong></td>
<td><strong>FINAL_LOCKED_POLICY</strong></td>
</tr>
<tr>
<td>Persistence</td>
<td style="text-align:right">66.836915</td>
<td style="text-align:right">26.737589</td>
<td style="text-align:right">21.513353</td>
<td style="text-align:right">0.459047</td>
<td>Task baseline</td>
</tr>
</tbody>
</table></div>
<p>Nguồn: <a class="rd-link" href="COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics_with_mape.json">Step 17 metrics with MAPE</a>, <a class="rd-link" href="COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_benchmark_signoff.json">Step 17 signoff</a> và <a class="rd-link" href="COURSE_WORK/artifacts/model_improvement_v2/model_improvement_v2_final_closure.json">V2 final closure</a>.</p>
<h3 class="rd-heading rd-h3" id="historical-v1-evidence">Historical V1 evidence</h3>
<p>V1 được giữ nguyên để làm baseline/provenance, không phải active final policy.</p>
<div class="rd-table-wrap"><table class="rd-table">
<thead>
<tr>
<th>Historical result</th>
<th style="text-align:right">RMSE (Wh)</th>
<th style="text-align:right">MAE (Wh)</th>
<th style="text-align:right">R²</th>
<th>Source</th>
</tr>
</thead>
<tbody>
<tr>
<td>V1 Transformer mean</td>
<td style="text-align:right">63.829658</td>
<td style="text-align:right">28.528603</td>
<td style="text-align:right">0.506422</td>
<td><a class="rd-link" href="COURSE_WORK/artifacts/final_test/phase_47_signoff.json">Phase 47 signoff</a></td>
</tr>
<tr>
<td>Persistence</td>
<td style="text-align:right">66.836915</td>
<td style="text-align:right">26.737589</td>
<td style="text-align:right">0.459047</td>
<td><a class="rd-link" href="COURSE_WORK/artifacts/final_test/final_test_summary.json">Final Test summary</a></td>
</tr>
</tbody>
</table></div>
<h3 class="rd-heading rd-h3" id="kết-quả-phân-tích-hiện-tại">Kết quả phân tích hiện tại</h3>
<ul>
<li>Phase 47–51 trình bày <code class="rd-inline-code rd-code-green">V2 FINAL</code>, được rebuild từ verified Step 17 metrics và frozen predictions.</li>
<li>Phase 52–57 giữ <code class="rd-inline-code rd-code-yellow">V1 HISTORICAL ATTENTION</code>; V2 Test attention tensors không được tạo, vì vậy không dùng các phase này để kết luận attention behavior của final V2.</li>
<li>Phase 58–59 trình bày <code class="rd-inline-code rd-code-green">V2 FINAL</code> summary và closure.</li>
<li>Step 14A chọn equal-weight E14-M1 seed ensemble; Step 14B không chọn persistence-neural blend.</li>
<li>E20 multi-seed finalist confirmation bị reject. Step 14A sau đó chọn fixed neural ensemble theo development-only evidence; Step 16 khóa và refit đúng ba seed trước Step 17.</li>
</ul>
</section><section class="rd-section rd-arch-card"><h2 class="rd-heading rd-h2" id="kiến-trúc-thư-mục"><span class="rd-section-pill">ARCHITECTURE</span>Kiến trúc thư mục</h2>
<p>Cấu trúc dưới đây phản ánh filesystem hiện tại của <code class="rd-inline-code">COURSE_WORK</code>:</p>
<pre class="rd-code"><code class="language-text">COURSE_WORK/
├── artifacts/
├── configs/
│   └── base/
├── docs/
│   ├── analysis_error/
│   ├── current_flow/
│   ├── link&amp;discussion_to_result/
│   ├── plan/
│   │   ├── plan_before_process/
│   │   ├── plan_detail_for_each_phase/
│   │   ├── plan_overview/
│   │   └── plan_to_refactor&amp;fix/
│   ├── rule_base/
│   └── save_log_in_processing/
├── link/
│   ├── interim/
│   └── raw_data/
├── notebook_course_work/
│   └── CourseWork.ipynb
├── scripts/
├── src/
│   └── course_work/
│       ├── analysis/
│       ├── attention/
│       ├── baselines/
│       ├── contracts/
│       ├── data/
│       ├── diagnostics/
│       ├── evaluation/
│       ├── experiments/
│       ├── final_model_lock/
│       ├── final_test_evaluation/
│       ├── lstm_tuning/
│       ├── metric_addendum/
│       ├── models/
│       ├── reporting/
│       ├── rolling_origin/
│       ├── sanity/
│       ├── scaling/
│       ├── scripts/
│       ├── sweeps/
│       ├── training/
│       ├── utils/
│       └── verification/
├── tests/
│   ├── contracts/
│   ├── integration/
│   └── unit/
├── pyproject.toml
└── requirements.txt
</code></pre>
<p>Một số tài liệu kiến trúc cũ vẫn mô tả dữ liệu dưới đường dẫn <code class="rd-inline-code">data/</code> và phạm vi phase trước khi dự án mở rộng. Filesystem hiện tại sử dụng <code class="rd-inline-code">link/raw_data</code> và <code class="rd-inline-code">link/interim</code>; README này lấy cấu trúc thực tế làm nguồn tham chiếu. Khi chỉnh sửa kiến trúc trong tương lai, cần đồng bộ lại tài liệu rule tương ứng thay vì tự động di chuyển dữ liệu trong một tác vụ tài liệu.</p>
</section><section class="rd-section rd-layers-card"><h2 class="rd-heading rd-h2" id="trách-nhiệm-của-từng-tầng"><span class="rd-section-pill">LAYERS</span>Trách nhiệm của từng tầng</h2>
<h3 class="rd-heading rd-h3" id="courseworkconfigs"><code class="rd-inline-code">COURSE_WORK/configs</code></h3>
<p>Chứa contract và cấu hình đầu vào ổn định. <code class="rd-inline-code">configs/base/coursework_contract.json</code> xác định bài toán, target, horizon, đơn vị, tiêu chí đánh giá, giao thức thời gian và các ràng buộc cốt lõi. Cấu hình phải được version hóa và fingerprint trước khi dùng làm đầu vào cho artifact.</p>
<h3 class="rd-heading rd-h3" id="courseworklink"><code class="rd-inline-code">COURSE_WORK/link</code></h3>
<ul>
<li><code class="rd-inline-code">raw_data</code>: dữ liệu nguồn, gói tải về, metadata, checksum và acquisition manifest.</li>
<li><code class="rd-inline-code">interim</code>: dữ liệu trung gian sau các bước làm sạch hoặc feature engineering nhưng trước những tầng downstream tương ứng.</li>
</ul>
<p>Tầng này chứa dữ liệu và metadata dữ liệu; không chứa logic huấn luyện hoặc code trực quan hóa.</p>
<h3 class="rd-heading rd-h3" id="courseworksrccourseworkdata"><code class="rd-inline-code">COURSE_WORK/src/course_work/data</code></h3>
<p>Chịu trách nhiệm cho acquisition, schema audit, temporal audit, feature engineering, feature-set registry, chronological split, window construction và DataLoader preparation. Mỗi module nên chỉ phục vụ một nhóm trách nhiệm dữ liệu rõ ràng.</p>
<h3 class="rd-heading rd-h3" id="courseworksrccourseworkscaling"><code class="rd-inline-code">COURSE_WORK/src/course_work/scaling</code></h3>
<p>Quản lý feature scaler và target scaler, đặc biệt là quy tắc fit trên Train. Scaler phải được lưu thành artifact kèm metadata để các phase sau tái sử dụng đúng phiên bản.</p>
<h3 class="rd-heading rd-h3" id="courseworksrccourseworkmodels"><code class="rd-inline-code">COURSE_WORK/src/course_work/models</code></h3>
<p>Chứa định nghĩa LSTM, Transformer và các component như attention, encoder layer, positional encoding, LoRA hoặc RevIN. Tầng model chỉ định nghĩa kiến trúc và forward behavior; không đảm nhiệm orchestration của sweep hoặc trình bày notebook.</p>
<h3 class="rd-heading rd-h3" id="courseworksrccourseworktraining"><code class="rd-inline-code">COURSE_WORK/src/course_work/training</code></h3>
<p>Chứa training engine, checkpointing, early stopping, pretraining, fine-tuning và reproducibility utilities. Tầng này nhận model, DataLoader và contract để thực hiện huấn luyện có thể tái lập.</p>
<h3 class="rd-heading rd-h3" id="courseworksrccourseworkevaluation"><code class="rd-inline-code">COURSE_WORK/src/course_work/evaluation</code></h3>
<p>Chứa metric dùng chung, inference và các hàm đánh giá. Metric phải được tính trên thang đo quy định, dùng cùng implementation giữa các mô hình và không tự ý mở Test.</p>
<h3 class="rd-heading rd-h3" id="courseworksrccourseworkexperiments"><code class="rd-inline-code">COURSE_WORK/src/course_work/experiments</code></h3>
<p>Quản lý registry, sweep orchestration, adaptation và final reference. Tầng này quyết định run nào cần thực thi hoặc có thể tái sử dụng dựa trên artifact hiện có.</p>
<h3 class="rd-heading rd-h3" id="courseworksrccourseworksweeps"><code class="rd-inline-code">COURSE_WORK/src/course_work/sweeps</code></h3>
<p>Mỗi phase sweep có module chuyên trách cho contract, prerequisite, condition evidence, training invocation, winner selection, signoff và reference update. Các sweep sử dụng Validation RMSE làm metric lựa chọn và giữ nguyên Test firewall.</p>
<h3 class="rd-heading rd-h3" id="courseworksrccourseworkanalysis"><code class="rd-inline-code">COURSE_WORK/src/course_work/analysis</code></h3>
<p>Chứa các phân tích sau khi mô hình đã được khóa và đánh giá: prediction, residual, error regime, worst error, attention extraction, heatmap, last-query, head comparison, error-conditioned attention, seed stability, final tables và final conclusions.</p>
<h3 class="rd-heading rd-h3" id="courseworksrccourseworkreporting"><code class="rd-inline-code">COURSE_WORK/src/course_work/reporting</code></h3>
<p>Chuyển artifact JSON hoặc bảng kết quả thành HTML tĩnh phù hợp với notebook. Reporting không huấn luyện model, không thay đổi artifact khoa học và không quyết định winner.</p>
<h3 class="rd-heading rd-h3" id="courseworksrccourseworkcontracts-và-verification"><code class="rd-inline-code">COURSE_WORK/src/course_work/contracts</code> và <code class="rd-inline-code">verification</code></h3>
<ul>
<li><code class="rd-inline-code">contracts</code>: định nghĩa và materialize các giao kèo giữa phase.</li>
<li><code class="rd-inline-code">verification</code>: kiểm tra checksum, schema, quan hệ phụ thuộc, trạng thái prerequisite và tính nhất quán của artifact.</li>
</ul>
<h3 class="rd-heading rd-h3" id="courseworkscripts"><code class="rd-inline-code">COURSE_WORK/scripts</code></h3>
<p>Chứa entry point chạy từ terminal, công cụ bảo trì, selective execution và recovery. Các tác vụ huấn luyện tốn tài nguyên được chạy tại đây hoặc qua module Python tương ứng, không đặt trực tiếp trong notebook.</p>
<h3 class="rd-heading rd-h3" id="courseworkartifacts"><code class="rd-inline-code">COURSE_WORK/artifacts</code></h3>
<p>Lưu đầu ra khoa học của từng phase, gồm JSON, CSV, model checkpoint, scaler, prediction bundle, figure, table, manifest, signoff và checksum. Đây là nguồn dữ liệu đầu vào cho các phase downstream và cho chế độ resume.</p>
<h3 class="rd-heading rd-h3" id="courseworkdocssaveloginprocessing"><code class="rd-inline-code">COURSE_WORK/docs/save_log_in_processing</code></h3>
<p>Lưu processing log rút gọn theo phase. Log cho biết phase đã chạy chưa, trạng thái, artifact liên quan, điều kiện thiếu và hành động tiếp theo. Log hỗ trợ notebook hiển thị kết quả mà không phải chạy lại toàn bộ pipeline.</p>
<h3 class="rd-heading rd-h3" id="courseworknotebookcoursework"><code class="rd-inline-code">COURSE_WORK/notebook_course_work</code></h3>
<p>Chứa notebook trình bày. Notebook gọi API từ package, đọc artifact hoặc processing log, và hiển thị HTML. Notebook không phải nơi đặt thuật toán xử lý dữ liệu, vòng lặp train, logic sweep hay quy tắc chọn winner.</p>
<h3 class="rd-heading rd-h3" id="courseworktests"><code class="rd-inline-code">COURSE_WORK/tests</code></h3>
<ul>
<li><code class="rd-inline-code">unit</code>: kiểm tra hàm và module độc lập.</li>
<li><code class="rd-inline-code">integration</code>: kiểm tra liên kết giữa các tầng và phase.</li>
<li><code class="rd-inline-code">contracts</code>: kiểm tra schema, invariant và giao kèo khoa học.</li>
</ul>
<h3 class="rd-heading rd-h3" id="courseworkdocs"><code class="rd-inline-code">COURSE_WORK/docs</code></h3>
<p>Chứa kế hoạch, phân tích lỗi, quy tắc kiến trúc, mô tả current flow, walkthrough notebook, kết quả thảo luận và hồ sơ refactor. Tài liệu phải phản ánh artifact đã tạo, không thay thế artifact khoa học.</p>
</section><section class="rd-section rd-flow-card"><h2 class="rd-heading rd-h2" id="luồng-xử-lý-đầu-cuối"><span class="rd-section-pill">END-TO-END FLOW</span>Luồng xử lý đầu cuối</h2>
<pre class="rd-code"><code class="language-mermaid">flowchart TD
    A[Coursework contract] --&gt; B[Environment verification]
    B --&gt; C[Data acquisition]
    C --&gt; D[Schema and temporal audits]
    D --&gt; E[Chronological split and EDA]
    E --&gt; F[Feature engineering and feature sets]
    F --&gt; G[Train-only scaling]
    G --&gt; H[Window populations and DataLoaders]
    H --&gt; I[Shared metrics and experiment registry]
    I --&gt; J[Persistence, LSTM and Transformer baselines]
    J --&gt; K[V1 sweeps and historical final evidence]
    K --&gt; L[MODEL_IMPROVEMENT_V2 E01-E20]
    L --&gt; M[Step 14A ensemble selection]
    M --&gt; N[Step 16 final refit and immutable lock]
    N --&gt; O[Step 17 POST_HOC_V2_BENCHMARK]
    O --&gt; P[V2 reporting-only analyses]
    P --&gt; Q[Phase 47-51 and 58-59 presentation]
</code></pre>
<p>Mỗi mũi tên biểu thị một quan hệ phụ thuộc có thể được kiểm tra bằng artifact, manifest, checksum hoặc signoff. Phase downstream không nên dựa vào biến còn tồn tại trong bộ nhớ notebook nếu cùng dữ liệu đã được materialize thành artifact.</p>
</section><section class="rd-section rd-phases-card"><h2 class="rd-heading rd-h2" id="chi-tiết-phase-059"><span class="rd-section-pill">60 PHASES</span>Chi tiết Phase 0–59</h2>
<h3 class="rd-heading rd-h3" id="phase-014-hợp-đồng-dữ-liệu-và-baseline-tối-thiểu">Phase 0–14: hợp đồng, dữ liệu và baseline tối thiểu</h3>
<div class="rd-table-wrap"><table class="rd-table">
<thead>
<tr>
<th style="text-align:right">Phase</th>
<th>Tên</th>
<th>Công việc chính</th>
<th>Đầu ra tiêu biểu</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right">0</td>
<td>Coursework Contract</td>
<td>Khóa bài toán, target, horizon, protocol, metric, câu hỏi nghiên cứu và leakage control</td>
<td>Contract JSON, fingerprint, signoff</td>
</tr>
<tr>
<td style="text-align:right">1</td>
<td>Environment</td>
<td>Kiểm tra Python, kernel, package, device và deterministic setup</td>
<td>Environment report, signoff</td>
</tr>
<tr>
<td style="text-align:right">2</td>
<td>Data Acquisition</td>
<td>Xác nhận nguồn, checksum, metadata và tệp dữ liệu gốc</td>
<td>Acquisition manifest, raw dataset evidence</td>
</tr>
<tr>
<td style="text-align:right">3</td>
<td>Schema Audit</td>
<td>Kiểm tra cột, kiểu dữ liệu, target, null, duplicate và miền giá trị</td>
<td>Schema manifest, discrepancy report</td>
</tr>
<tr>
<td style="text-align:right">4</td>
<td>Temporal Integrity Audit</td>
<td>Kiểm tra timestamp, thứ tự, tần suất và khoảng trống thời gian</td>
<td>Temporal manifest, signoff</td>
</tr>
<tr>
<td style="text-align:right">5</td>
<td>Chronological Split</td>
<td>Gán quan sát vào Train, Validation và Test theo thời gian</td>
<td>Split manifest, membership và fingerprints</td>
</tr>
<tr>
<td style="text-align:right">6</td>
<td>Exploratory Data Analysis</td>
<td>Khảo sát phân phối, thống kê, tương quan và hành vi theo thời gian</td>
<td>EDA tables, figures, report</td>
</tr>
<tr>
<td style="text-align:right">7</td>
<td>Feature Engineering</td>
<td>Tạo time feature và đặc trưng hợp lệ theo thời gian</td>
<td>Feature-engineered data, feature manifest</td>
</tr>
<tr>
<td style="text-align:right">8</td>
<td>Feature-Set Variants</td>
<td>Đăng ký các nhóm feature dùng cho thí nghiệm</td>
<td>Feature-set registry, signoff</td>
</tr>
<tr>
<td style="text-align:right">9</td>
<td>Train-Only Scaling</td>
<td>Fit feature và target scaler trên Train</td>
<td>Scaler artifacts, scaling manifest</td>
</tr>
<tr>
<td style="text-align:right">10</td>
<td>Window Builder</td>
<td>Tạo cửa sổ sequence-to-one theo lookback và horizon</td>
<td>Window populations, boundary metadata</td>
</tr>
<tr>
<td style="text-align:right">11</td>
<td>DataLoaders</td>
<td>Đóng gói tensor và DataLoader theo partition</td>
<td>Loader manifest, batch evidence</td>
</tr>
<tr>
<td style="text-align:right">12</td>
<td>Shared Metrics</td>
<td>Chuẩn hóa MAE, RMSE, R² và contract metric dùng chung</td>
<td>Metric contract, verification output</td>
</tr>
<tr>
<td style="text-align:right">13</td>
<td>Experiment Registry</td>
<td>Đăng ký run, cấu hình, seed và quan hệ artifact</td>
<td>Experiment registry, run identity</td>
</tr>
<tr>
<td style="text-align:right">14</td>
<td>Persistence Baseline</td>
<td>Đánh giá dự báo <code class="rd-inline-code">y(t+1)=y(t)</code> trên Validation</td>
<td>Baseline metrics, predictions, signoff</td>
</tr>
</tbody>
</table></div>
<p>Phase 0 là contract nội bộ và có thể được ẩn khỏi phần trình bày chính của notebook, nhưng vẫn là dependency khoa học của toàn pipeline.</p>
<h3 class="rd-heading rd-h3" id="phase-1522-mô-hình-và-baseline-học-sâu">Phase 15–22: mô hình và baseline học sâu</h3>
<div class="rd-table-wrap"><table class="rd-table">
<thead>
<tr>
<th style="text-align:right">Phase</th>
<th>Tên</th>
<th>Công việc chính</th>
<th>Đầu ra tiêu biểu</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right">15</td>
<td>LSTM Implementation</td>
<td>Xây dựng LSTM regression theo input contract</td>
<td>Model definition, shape checks</td>
</tr>
<tr>
<td style="text-align:right">16</td>
<td>Transformer Implementation</td>
<td>Xây dựng Transformer Encoder regression</td>
<td>Model definition, configuration contract</td>
</tr>
<tr>
<td style="text-align:right">17</td>
<td>Attention-Aware Encoder Verification</td>
<td>Xác minh attention có thể được trích xuất đúng hình dạng</td>
<td>Attention verification artifact</td>
</tr>
<tr>
<td style="text-align:right">18</td>
<td>Forward-Pass Sanity Tests</td>
<td>Kiểm tra shape, finite output và forward behavior</td>
<td>Sanity signoff</td>
</tr>
<tr>
<td style="text-align:right">19</td>
<td>Baseline Training Engine</td>
<td>Chuẩn hóa train loop, validation, checkpoint và early stopping</td>
<td>Engine verification, training contract</td>
</tr>
<tr>
<td style="text-align:right">20</td>
<td>LSTM Baseline Run</td>
<td>Huấn luyện và đánh giá LSTM trên Validation</td>
<td>Checkpoint, history, metrics, signoff</td>
</tr>
<tr>
<td style="text-align:right">21</td>
<td>Transformer B0 Run</td>
<td>Huấn luyện Transformer cấu hình gốc</td>
<td>Checkpoint, history, metrics, signoff</td>
</tr>
<tr>
<td style="text-align:right">22</td>
<td>Learning-Curve Diagnostics</td>
<td>Phân tích train/validation curves và dấu hiệu underfit hoặc overfit</td>
<td>Diagnostic report, curves</td>
</tr>
</tbody>
</table></div>
<h3 class="rd-heading rd-h3" id="phase-2341-chuỗi-sweep-s1s19">Phase 23–41: chuỗi sweep S1–S19</h3>
<div class="rd-table-wrap"><table class="rd-table">
<thead>
<tr>
<th style="text-align:right">Phase</th>
<th>Sweep</th>
<th>Yếu tố thay đổi</th>
<th>Mục đích</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right">23</td>
<td>S1</td>
<td>Feature set</td>
<td>Chọn nhóm biến đầu vào</td>
</tr>
<tr>
<td style="text-align:right">24</td>
<td>S2</td>
<td>Time feature</td>
<td>Chọn biểu diễn thời gian</td>
</tr>
<tr>
<td style="text-align:right">25</td>
<td>S3</td>
<td>Target scaling</td>
<td>Chọn cách chuẩn hóa target</td>
</tr>
<tr>
<td style="text-align:right">26</td>
<td>S4</td>
<td>Lookback</td>
<td>Chọn chiều dài lịch sử</td>
</tr>
<tr>
<td style="text-align:right">27</td>
<td>S5</td>
<td>Pooling</td>
<td>Chọn cách gom representation theo thời gian</td>
</tr>
<tr>
<td style="text-align:right">28</td>
<td>S6</td>
<td>Activation</td>
<td>Chọn activation của encoder</td>
</tr>
<tr>
<td style="text-align:right">29</td>
<td>S7</td>
<td>Batch size</td>
<td>Chọn kích thước batch</td>
</tr>
<tr>
<td style="text-align:right">30</td>
<td>S8</td>
<td>Learning rate</td>
<td>Chọn tốc độ học AdamW</td>
</tr>
<tr>
<td style="text-align:right">31</td>
<td>S9</td>
<td>Weight decay</td>
<td>Chọn mức regularization của AdamW</td>
</tr>
<tr>
<td style="text-align:right">32</td>
<td>S10</td>
<td>Dropout</td>
<td>Chọn dropout tại các vị trí encoder</td>
</tr>
<tr>
<td style="text-align:right">33</td>
<td>S11</td>
<td><code class="rd-inline-code">d_model</code></td>
<td>Chọn chiều rộng embedding Transformer</td>
</tr>
<tr>
<td style="text-align:right">34</td>
<td>S12</td>
<td>Attention heads</td>
<td>Chọn số multi-head attention head</td>
</tr>
<tr>
<td style="text-align:right">35</td>
<td>S13</td>
<td>Encoder layers</td>
<td>Chọn độ sâu encoder</td>
</tr>
<tr>
<td style="text-align:right">36</td>
<td>S14</td>
<td>FFN width</td>
<td>Chọn chiều rộng feed-forward network</td>
</tr>
<tr>
<td style="text-align:right">37</td>
<td>S15</td>
<td>Loss</td>
<td>So sánh hàm loss huấn luyện</td>
</tr>
<tr>
<td style="text-align:right">38</td>
<td>S16</td>
<td>Epoch cap</td>
<td>Chọn giới hạn epoch phù hợp</td>
</tr>
<tr>
<td style="text-align:right">39</td>
<td>S17</td>
<td>Gradient clipping</td>
<td>Đánh giá ngưỡng clipping</td>
</tr>
<tr>
<td style="text-align:right">40</td>
<td>S18</td>
<td>RevIN</td>
<td>Đánh giá normalization đảo ngược</td>
</tr>
<tr>
<td style="text-align:right">41</td>
<td>S19</td>
<td>Boundary protocol</td>
<td>So sánh WB0 và WB1, kiểm tra tính hợp lệ biên</td>
</tr>
</tbody>
</table></div>
<p>Mỗi sweep tạo condition evidence, results, winner, reference update và phase signoff. Khi artifact hợp lệ đã tồn tại, phase có thể tái sử dụng kết quả thay vì train lại.</p>
<h3 class="rd-heading rd-h3" id="phase-4247-historical-v1-tổng-hợp-khóa-mô-hình-và-test">Phase 42–47: historical V1 tổng hợp, khóa mô hình và Test</h3>
<div class="rd-table-wrap"><table class="rd-table">
<thead>
<tr>
<th style="text-align:right">Phase</th>
<th>Tên</th>
<th>Scientific evidence</th>
<th>Presentation hiện tại</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right">42</td>
<td>Candidate Synthesis</td>
<td>V1 historical</td>
<td>V1 historical</td>
</tr>
<tr>
<td style="text-align:right">43</td>
<td>LSTM Tuning</td>
<td>V1 historical</td>
<td>V1 historical</td>
</tr>
<tr>
<td style="text-align:right">44</td>
<td>Rolling-Origin Robustness</td>
<td>V1 historical</td>
<td>V1 historical</td>
</tr>
<tr>
<td style="text-align:right">45</td>
<td>Final Model Lock</td>
<td>V1 historical</td>
<td>V1 historical</td>
</tr>
<tr>
<td style="text-align:right">46</td>
<td>Three-Seed Final Runs</td>
<td>V1 historical</td>
<td>V1 historical</td>
</tr>
<tr>
<td style="text-align:right">47</td>
<td>Final Test Evaluation</td>
<td>Historical V1 artifacts giữ nguyên</td>
<td><code class="rd-inline-code rd-code-green">V2 FINAL</code> từ Step 17 verified artifact</td>
</tr>
</tbody>
</table></div>
<p>Phase 47 presentation không overwrite hoặc đổi nhãn V1 artifacts; nó hiển thị final V2 benchmark cho báo cáo cuối, trong khi historical V1 Phase 47 vẫn là provenance.</p>
<h3 class="rd-heading rd-h3" id="phase-4859-phân-tích-và-kết-luận">Phase 48–59: phân tích và kết luận</h3>
<div class="rd-table-wrap"><table class="rd-table">
<thead>
<tr>
<th style="text-align:right">Phase</th>
<th>Tên</th>
<th>Công việc chính</th>
<th>Đầu ra tiêu biểu</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right">48</td>
<td>Prediction Analysis</td>
<td>V2 reporting từ frozen Step 17 predictions</td>
<td>V2 tables/figures</td>
</tr>
<tr>
<td style="text-align:right">49</td>
<td>Residual Analysis</td>
<td>V2 reporting từ frozen Step 17 predictions</td>
<td>V2 residual summaries</td>
</tr>
<tr>
<td style="text-align:right">50</td>
<td>Error-by-Regime Analysis</td>
<td>V2 reporting trên verified matched population</td>
<td>V2 regime tables</td>
</tr>
<tr>
<td style="text-align:right">51</td>
<td>Worst-Error Analysis</td>
<td>V2 reporting từ frozen predictions</td>
<td>V2 worst-case tables</td>
</tr>
<tr>
<td style="text-align:right">52</td>
<td>Attention Extraction</td>
<td>Historical V1 attention evidence</td>
<td>V1 historical artifacts</td>
</tr>
<tr>
<td style="text-align:right">53</td>
<td>Attention Heatmaps</td>
<td>Historical V1 attention evidence</td>
<td>V1 historical figures</td>
</tr>
<tr>
<td style="text-align:right">54</td>
<td>Last-Query Attention</td>
<td>Historical V1 attention evidence</td>
<td>V1 historical summary</td>
</tr>
<tr>
<td style="text-align:right">55</td>
<td>Head Comparison</td>
<td>Historical V1 attention evidence</td>
<td>V1 historical comparison</td>
</tr>
<tr>
<td style="text-align:right">56</td>
<td>Error-Conditioned Attention</td>
<td>Historical V1 attention evidence</td>
<td>V1 historical summary</td>
</tr>
<tr>
<td style="text-align:right">57</td>
<td>Seed-Stability Attention Check</td>
<td>Historical V1 attention evidence</td>
<td>V1 historical signoff</td>
</tr>
<tr>
<td style="text-align:right">58</td>
<td>Final Results Summary</td>
<td>Verified V2 lock, Step 17, MAPE và reporting analysis</td>
<td>V2 final tables</td>
</tr>
<tr>
<td style="text-align:right">59</td>
<td>Final Conclusions</td>
<td>Verified V2 final closure</td>
<td>V2 final conclusions</td>
</tr>
</tbody>
</table></div>
<h3 class="rd-heading rd-h3" id="modelimprovementv2-e01-đến-step-17">MODEL_IMPROVEMENT_V2 — E01 đến Step 17</h3>
<div class="rd-table-wrap"><table class="rd-table">
<thead>
<tr>
<th>Giai đoạn</th>
<th>Vai trò</th>
<th>Kết quả khóa</th>
</tr>
</thead>
<tbody>
<tr>
<td>E01–E20</td>
<td>Development experiments và matched multi-seed confirmation</td>
<td>Không dùng Test để tune/chọn</td>
</tr>
<tr>
<td>Step 14A</td>
<td>Equal-weight seed ensemble</td>
<td>Chọn <code class="rd-inline-code">mean(E14-M1 seeds 42,123,2026)</code></td>
</tr>
<tr>
<td>Step 14B</td>
<td>Fixed persistence-neural blends</td>
<td>Giữ neural ensemble</td>
</tr>
<tr>
<td>Step 16</td>
<td>Full pre-Test final refit + immutable lock</td>
<td>3 seeds, 16 epochs/seed, equal weights</td>
</tr>
<tr>
<td>Step 17</td>
<td>Post-hoc benchmark</td>
<td><code class="rd-inline-code rd-code-yellow">POST_HOC_V2_BENCHMARK</code>; không retune</td>
</tr>
</tbody>
</table></div>
</section><section class="rd-section rd-resume-card"><h2 class="rd-heading rd-h2" id="cơ-chế-artifact-log-và-selective-resume"><span class="rd-section-pill">RESUME &amp; ARTIFACTS</span>Cơ chế artifact, log và selective resume</h2>
<h3 class="rd-heading rd-h3" id="vì-sao-không-chạy-lại-toàn-bộ-notebook">Vì sao không chạy lại toàn bộ notebook</h3>
<p>Các phase huấn luyện và sweep có thể tốn nhiều thời gian và tài nguyên. Dự án materialize kết quả theo phase để một phase downstream có thể đọc trạng thái đã lưu, kiểm tra đủ điều kiện và chỉ thực thi phần còn thiếu.</p>
<h3 class="rd-heading rd-h3" id="các-lớp-trạng-thái">Các lớp trạng thái</h3>
<ol>
<li><code class="rd-inline-code">artifacts</code>: nguồn bằng chứng khoa học đầy đủ, gồm kết quả, model, scaler, prediction và signoff.</li>
<li><code class="rd-inline-code">docs/save_log_in_processing</code>: log trình bày và trạng thái rút gọn cho từng phase.</li>
<li>Experiment registry: định danh run, cấu hình và quan hệ giữa run với artifact.</li>
<li>Checksum và fingerprint: xác minh artifact chưa bị thay đổi ngoài quy trình.</li>
<li>Prerequisite resolver: quyết định <code class="rd-inline-code">REUSE</code>, <code class="rd-inline-code">RUN_MISSING</code>, <code class="rd-inline-code">REFRESH</code> hoặc <code class="rd-inline-code">BLOCK</code> tùy trạng thái đầu vào.</li>
</ol>
<h3 class="rd-heading rd-h3" id="quy-trình-selective-resume">Quy trình selective resume</h3>
<pre class="rd-code"><code class="language-text">Chọn phase cần tiếp tục
        |
        v
Đọc processing log và signoff đã có
        |
        v
Xác minh prerequisite, checksum và environment
        |
        +--&gt; Hợp lệ và đầy đủ: tái sử dụng artifact
        |
        +--&gt; Thiếu condition: chỉ chạy condition còn thiếu
        |
        +--&gt; Artifact stale nhưng có thể tái tạo: refresh đúng phạm vi
        |
        +--&gt; Upstream không hợp lệ hoặc Test firewall vi phạm: block
        |
        v
Cập nhật artifact, signoff và processing log
        |
        v
Notebook đọc JSON và render HTML tĩnh
</code></pre>
<p>Trạng thái như <code class="rd-inline-code rd-code-blue">VALID_REUSABLE</code> mô tả khả năng tái sử dụng artifact, không phải lỗi và cũng không đồng nghĩa với <code class="rd-inline-code">BLOCKED</code>. Reporting layer cần chuẩn hóa trạng thái này thành thông tin dễ hiểu trong overview mà không thay đổi trạng thái khoa học gốc.</p>
<h3 class="rd-heading rd-h3" id="nguyên-tắc-sử-dụng">Nguyên tắc sử dụng</h3>
<ul>
<li>Không xem output còn trong kernel là nguồn sự thật.</li>
<li>Không bỏ qua prerequisite chỉ vì file đích đã tồn tại.</li>
<li>Không sửa thủ công winner hoặc signoff để vượt qua trạng thái block.</li>
<li>Không chạy lại Test khi chưa có lý do được protocol cho phép.</li>
<li>Khi environment khác với environment đã ghi, phải xác minh compatibility trước khi tái sử dụng checkpoint hoặc artifact nhạy cảm.</li>
</ul>
</section><section class="rd-section rd-notebook-card"><h2 class="rd-heading rd-h2" id="vai-trò-của-notebook"><span class="rd-section-pill">NOTEBOOK</span>Vai trò của notebook</h2>
<p><a class="rd-link rd-notebook-link" href="COURSE_WORK/notebook_course_work/CourseWork.ipynb" title="Opens CourseWork.ipynb. Markdown links do not reliably deep-link to a specific notebook cell.">CourseWork.ipynb</a> hiện có <code class="rd-inline-code">153</code> cells và là giao diện trình bày của pipeline.</p>
<p>Notebook được phép:</p>
<ul>
<li>Import API công khai từ <code class="rd-inline-code">course_work</code>.</li>
<li>Gọi entry point materialize hoặc render của phase.</li>
<li>Đọc JSON, CSV, figure và metadata đã được tạo.</li>
<li>Hiển thị bảng, biểu đồ và HTML tĩnh.</li>
<li>Trình bày định nghĩa bài toán, quyết định thực nghiệm và kết quả.</li>
</ul>
<p>Notebook không nên:</p>
<ul>
<li>Chứa vòng lặp huấn luyện hoặc logic tối ưu siêu tham số.</li>
<li>Tự xây dựng feature, scaler, split hoặc window bằng code nghiệp vụ nội tuyến.</li>
<li>Tự chọn winner bằng logic chỉ tồn tại trong cell.</li>
<li>Phụ thuộc vào thứ tự chạy cell để giữ dữ liệu quan trọng trong bộ nhớ.</li>
<li>Dùng widget state làm nguồn duy nhất cho kết quả đã lưu.</li>
<li>Chạy lại toàn pipeline khi chỉ cần render một phase từ artifact.</li>
</ul>
<p>HTML được tạo ở reporting layer phải là output tĩnh có thể hiển thị ổn định trong Jupyter và VS Code. Cách này tránh lỗi mất widget model khi notebook được mở ở môi trường khác.</p>
</section><section class="rd-section rd-setup-card"><h2 class="rd-heading rd-h2" id="cài-đặt-và-sử-dụng"><span class="rd-section-pill">GETTING STARTED</span>Cài đặt và sử dụng</h2>
<h3 class="rd-heading rd-h3" id="yêu-cầu-môi-trường">Yêu cầu môi trường</h3>
<ul>
<li>Python từ 3.10 và nhỏ hơn 3.11.</li>
<li>Môi trường ảo riêng cho coursework.</li>
<li>Các dependency theo <code class="rd-inline-code">requirements.txt</code> hoặc metadata trong <code class="rd-inline-code">pyproject.toml</code>.</li>
</ul>
<h3 class="rd-heading rd-h3" id="tạo-môi-trường">Tạo môi trường</h3>
<p>Từ thư mục gốc repository:</p>
<pre class="rd-code"><code class="language-bash">cd COURSE_WORK
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
</code></pre>
<p>Editable install giúp notebook và script import <code class="rd-inline-code">course_work</code> từ <code class="rd-inline-code">src/course_work</code> mà không chèn đường dẫn thủ công vào từng cell.</p>
<h3 class="rd-heading rd-h3" id="mở-notebook">Mở notebook</h3>
<pre class="rd-code"><code class="language-bash">cd COURSE_WORK
jupyter notebook notebook_course_work/CourseWork.ipynb
</code></pre>
<p>Chọn đúng kernel thuộc <code class="rd-inline-code">.venv</code>. Trước khi chạy một phase, đọc phần prerequisite và trạng thái artifact của phase đó. Với phase đã materialize, ưu tiên cell render từ JSON thay vì chạy lại training.</p>
<h3 class="rd-heading rd-h3" id="chạy-module-hoặc-script-từ-terminal">Chạy module hoặc script từ terminal</h3>
<p>Các phase tốn tài nguyên phải được chạy bằng entry point hiện có trong <code class="rd-inline-code">COURSE_WORK/scripts</code> hoặc module tương ứng dưới <code class="rd-inline-code">src/course_work</code>. Trước khi chạy:</p>
<ol>
<li>Kích hoạt đúng virtual environment.</li>
<li>Đứng tại thư mục <code class="rd-inline-code">COURSE_WORK</code>.</li>
<li>Kiểm tra processing log của phase trước.</li>
<li>Kiểm tra artifact và signoff prerequisite.</li>
<li>Chỉ chạy phase hoặc condition cần thiết.</li>
<li>Xác minh output, checksum, signoff và log ngay sau khi hoàn thành.</li>
</ol>
<p>Tên lệnh cụ thể phụ thuộc phase. Không suy đoán tên script từ tên phase; đối chiếu <a class="rd-link" href="COURSE_WORK/docs/link&amp;discussion_to_result/NOTEBOOK_CELLS_WALKTHROUGH.md">walkthrough notebook</a>, kế hoạch phase và nội dung <code class="rd-inline-code">COURSE_WORK/scripts</code> trước khi thực thi.</p>
</section><section class="rd-section rd-tests-card"><h2 class="rd-heading rd-h2" id="kiểm-thử-và-tái-lập"><span class="rd-section-pill">TESTING</span>Kiểm thử và tái lập</h2>
<h3 class="rd-heading rd-h3" id="chạy-test">Chạy test</h3>
<pre class="rd-code"><code class="language-bash">cd COURSE_WORK
source .venv/bin/activate
python -m pytest tests
</code></pre>
<p>Có thể chạy test theo tầng:</p>
<pre class="rd-code"><code class="language-bash">python -m pytest tests/unit
python -m pytest tests/integration
python -m pytest tests/contracts
</code></pre>
<h3 class="rd-heading rd-h3" id="kiểm-tra-tối-thiểu-sau-mỗi-thay-đổi">Kiểm tra tối thiểu sau mỗi thay đổi</h3>
<ul>
<li>Import module đã sửa.</li>
<li>Chạy unit test liên quan trực tiếp.</li>
<li>Chạy contract hoặc integration test của phase.</li>
<li>Xác minh upstream artifact vẫn hợp lệ.</li>
<li>Xác minh phase signoff không báo discrepancy mới.</li>
<li>Kiểm tra notebook render được artifact mới mà không huấn luyện lại.</li>
<li>Kiểm tra Test firewall nếu thay đổi liên quan split, scaler, window, metric hoặc evaluation.</li>
</ul>
<h3 class="rd-heading rd-h3" id="tái-lập-kết-quả">Tái lập kết quả</h3>
<p>Khả năng tái lập dựa trên:</p>
<ul>
<li>Environment report và phiên bản dependency.</li>
<li>Fixed seed và deterministic configuration khi được hỗ trợ.</li>
<li>Dataset checksum và manifest.</li>
<li>Config fingerprint.</li>
<li>Train-only scaler artifact.</li>
<li>Window population identity.</li>
<li>Run ID, checkpoint metadata và experiment registry.</li>
<li>Phase signoff và checksum đầu vào/đầu ra.</li>
<li>Final model lock trước Test.</li>
</ul>
<p>Phase 59 ghi nhận 49 acceptance check của chính phase này đều đạt. Con số này mô tả bộ acceptance của Phase 59, không được hiểu là tổng số test hiện có trong toàn repository.</p>
</section><section class="rd-section rd-docs-card"><h2 class="rd-heading rd-h2" id="tài-liệu-dự-án"><span class="rd-section-pill">DOCUMENTATION</span>Tài liệu dự án</h2>
<h3 class="rd-heading rd-h3" id="quy-tắc-và-kiến-trúc">Quy tắc và kiến trúc</h3>
<ul>
<li><a class="rd-link" href="working_rule.md">Quy tắc làm việc chung</a></li>
<li><a class="rd-link" href="COURSE_WORK/docs/rule_base/architecture_rule.md">Quy tắc kiến trúc</a></li>
<li><a class="rd-link" href="COURSE_WORK/docs/rule_base/rule_code.md">Quy tắc code</a></li>
</ul>
<h3 class="rd-heading rd-h3" id="kế-hoạch-và-phân-tích-yêu-cầu">Kế hoạch và phân tích yêu cầu</h3>
<ul>
<li><a class="rd-link" href="COURSE_WORK/docs/plan/plan_overview/Requirement_analyst_&amp;_plan_to_make.md">Phân tích yêu cầu và kế hoạch</a></li>
<li><a class="rd-link" href="COURSE_WORK/docs/plan/plan_overview/Main_plan.md">Kế hoạch chính</a></li>
<li><a class="rd-link" href="COURSE_WORK/docs/plan/plan_detail_for_each_phase">Kế hoạch chi tiết từng phase</a></li>
<li><a class="rd-link" href="COURSE_WORK/docs/analysis_error">Phân tích lỗi</a></li>
<li><a class="rd-link" href="COURSE_WORK/docs/plan/plan_before_process">Kế hoạch trước xử lý</a></li>
<li><a class="rd-link" href="COURSE_WORK/docs/plan/plan_to_refactor&amp;fix">Hồ sơ refactor và fix</a></li>
</ul>
<h3 class="rd-heading rd-h3" id="current-flow-và-kết-quả">Current flow và kết quả</h3>
<ul>
<li><a class="rd-link" href="COURSE_WORK/docs/current_flow/CURRENT_FLOW_SUMMARY.md">Tổng hợp current flow</a></li>
<li><a class="rd-link" href="COURSE_WORK/docs/current_flow/PHASE_1_TO_33_CURRENT_FLOW_DETAIL.md">Chi tiết Phase 1–33</a></li>
<li><a class="rd-link" href="COURSE_WORK/docs/current_flow/PHASE_34_TO_59_CURRENT_FLOW_DETAIL.md">Chi tiết Phase 34–59</a></li>
<li><a class="rd-link" href="COURSE_WORK/docs/link&amp;discussion_to_result/NOTEBOOK_CELLS_WALKTHROUGH.md">Walkthrough notebook và liên kết cell</a></li>
<li><a class="rd-link" href="COURSE_WORK/artifacts/final_conclusions/FINAL_PROJECT_SUMMARY.md">Tóm tắt dự án cuối</a></li>
</ul>
</section><section class="rd-section rd-complete-card"><h2 class="rd-heading rd-h2" id="những-gì-dự-án-đã-hoàn-thành"><span class="rd-section-pill">COMPLETED</span>Những gì dự án đã hoàn thành</h2>
<ul>
<li>Xây dựng pipeline đầy đủ từ contract đến kết luận cuối cho 60 phase đánh số 0–59.</li>
<li>Audit schema và tính toàn vẹn thời gian trước khi huấn luyện.</li>
<li>Triển khai chronological split, Train-only scaling và window protocol có kiểm soát leakage.</li>
<li>Xây dựng persistence baseline, LSTM baseline và Transformer Encoder regression.</li>
<li>Chuẩn hóa training engine, metric, registry, checkpoint và reproducibility metadata.</li>
<li>Thực hiện chuỗi 19 sweep từ feature set đến boundary protocol.</li>
<li>Tổng hợp candidate, đánh giá rolling-origin và khóa cấu hình cuối trước Test.</li>
<li>Refit Transformer với ba seed và thực hiện Final Test evaluation theo population khóa.</li>
<li>Bổ sung MAPE có safe policy mà không thay đổi metric lựa chọn RMSE.</li>
<li>Thực hiện prediction analysis, residual analysis, error-by-regime và worst-error audit.</li>
<li>Trích xuất và phân tích attention theo heatmap, last query, head, mức sai số và seed.</li>
<li>Tạo final tables, project summary, conclusion signoff và evidence map.</li>
<li>Tách logic xử lý khỏi notebook, dùng artifact JSON và HTML reporting để hỗ trợ selective resume.</li>
</ul>
</section><section class="rd-section rd-limit-card"><h2 class="rd-heading rd-h2" id="giới-hạn-và-hướng-phát-triển"><span class="rd-section-pill">LIMITATIONS</span>Giới hạn và hướng phát triển</h2>
<h3 class="rd-heading rd-h3" id="giới-hạn-hiện-tại">Giới hạn hiện tại</h3>
<ul>
<li>Dữ liệu đến từ một hộ gia đình nên chưa chứng minh được khả năng tổng quát hóa rộng.</li>
<li>Bài toán chỉ dự báo một bước 10 phút; chưa đánh giá multi-horizon forecasting.</li>
<li>Chuỗi sweep tuần tự hiệu quả về chi phí nhưng không khám phá đầy đủ tương tác giữa mọi hyperparameter.</li>
<li>Ba seed cung cấp ước lượng độ ổn định ban đầu nhưng chưa phải phân tích bất định quy mô lớn.</li>
<li>Dữ liệu chuỗi thời gian có phụ thuộc mạnh nên không thể diễn giải các mẫu như quan sát độc lập.</li>
<li>Attention là bằng chứng mô tả hành vi mô hình, không phải giải thích nhân quả.</li>
<li>LSTM và Transformer cuối không dùng cùng lookback population, vì vậy chưa có so sánh LSTM–Transformer cuối trên cùng <code class="rd-inline-code">FINAL_TEST_POP-v1</code>.</li>
<li>Test MAPE đã được bổ sung từ frozen Step 17 prediction artifacts tại <a class="rd-link" href="COURSE_WORK/artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics_with_mape.json"><code class="rd-inline-code">step17_metrics_with_mape.json</code></a>; addendum không train, không inference mới và không mở lại raw Test source.</li>
</ul>
<h3 class="rd-heading rd-h3" id="hướng-phát-triển-phù-hợp">Hướng phát triển phù hợp</h3>
<ul>
<li>Đánh giá multi-horizon và direct multi-output forecasting.</li>
<li>Thực hiện nested hoặc blocked time-series validation khi ngân sách cho phép.</li>
<li>So sánh các mô hình time-series hiện đại khác trên cùng protocol và cùng population.</li>
<li>Mở rộng sang nhiều hộ gia đình hoặc domain khác để kiểm tra external validity.</li>
<li>Nghiên cứu uncertainty estimation và prediction interval.</li>
<li>Thêm kiểm tra drift và quy trình inference phục vụ triển khai.</li>
<li>Tạo prediction bundle chuẩn ngay tại Final Test để mọi metric bổ sung có thể được tính mà không mở lại Test.</li>
<li>Đồng bộ toàn bộ tài liệu kiến trúc cũ với filesystem hiện tại và phạm vi Phase 0–59 trong một tác vụ quản trị tài liệu riêng.</li>
</ul>
</section><section class="rd-section rd-status-card"><h2 class="rd-heading rd-h2" id="trạng-thái-dự-án"><span class="rd-section-pill">STATUS</span>Trạng thái dự án</h2>
<div class="rd-table-wrap"><table class="rd-table">
<thead>
<tr>
<th>Thành phần</th>
<th>Trạng thái hiện tại</th>
</tr>
</thead>
<tbody>
<tr>
<td>Final V2 lock</td>
<td><code class="rd-inline-code rd-code-green">LOCKED</code></td>
</tr>
<tr>
<td>Final policy</td>
<td><code class="rd-inline-code rd-code-green">V2 Equal-weight Ensemble = FINAL_LOCKED_POLICY</code></td>
</tr>
<tr>
<td>Step 17</td>
<td><code class="rd-inline-code rd-code-yellow">POST_HOC_V2_BENCHMARK</code> hoàn tất</td>
</tr>
<tr>
<td>MODEL_IMPROVEMENT_V2</td>
<td><code class="rd-inline-code rd-code-green">COMPLETE</code></td>
</tr>
<tr>
<td>Post-Test retuning</td>
<td><code class="rd-inline-code rd-code-green">FALSE</code></td>
</tr>
<tr>
<td>V1 evidence</td>
<td>Historical baseline/provenance, giữ nguyên</td>
</tr>
</tbody>
</table></div>
<p>Luồng khoa học Phase 0–59 đã hoàn tất và kết luận cuối đã được khóa trong artifact Phase 59. Mọi thay đổi tiếp theo có khả năng ảnh hưởng đến kết quả khoa học phải bắt đầu bằng phân tích phạm vi, kế hoạch được duyệt, kiểm tra dependency và một phiên bản artifact mới; không ghi đè im lặng lên kết quả đã khóa.</p>
</section></div>
<div class="rd-footer"><h3>COURSE_WORK · Scientific Pipeline Locked</h3><p>Contract → Data Integrity → Models → Controlled Sweeps → MODEL_IMPROVEMENT_V2 → Step 16 Immutable Lock → Step 17 Post-hoc Benchmark → Final Reporting</p><div class="rd-floaters"><span class="rd-floater">🐍</span><span class="rd-floater">🔥</span><span class="rd-floater">⚡</span><span class="rd-floater">📈</span><span class="rd-floater">✓</span></div><svg class="rd-wave" viewBox="0 0 1200 120" preserveAspectRatio="none" aria-hidden="true"><path d="M0,60 C180,120 320,0 520,55 C720,110 900,10 1200,60 L1200,120 L0,120 Z" fill="#4cc9f0" opacity=".48"/><path d="M0,78 C220,25 380,115 610,66 C840,18 1010,100 1200,54 L1200,120 L0,120 Z" fill="#f5c84b" opacity=".46"/></svg></div>
</div>
