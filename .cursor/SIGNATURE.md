# AI Agent Signature — `.cursor/`

**Repository:** `/Users/ticoder-coder/Documents/SGOD/SAM-V2`

**Agent:** All AI Agents (Main Agent + Subagents — bao gồm Cursor IDE AI, Cursor Composer, Subagents)

**Folder:** `/Users/ticoder-coder/Documents/SGOD/SAM-V2/.cursor/`

**Acknowledgement date:** `2026-06-15`

---

## 📂 Rule Sources Acknowledged

### 📁 `.cursor/agents/`

| File | Mô tả |
|---|---|
| `ci-monitor-subagent.md` | **CI Monitor Subagent** — helper cho `/monitor-ci` command. Gọi **MCP tool** `ci_information` hoặc `update_self_healing_fix` một lần và trả về kết quả. **KHÔNG loop, poll, sleep, hay tự quyết định.** Các command: `FETCH_STATUS`, `FETCH_HEAVY`, `UPDATE_FIX`, `FETCH_THROTTLE_INFO`. Chỉ extract và return **CÁC FIELD ĐƯỢC CHỈ ĐỊNH** — không dump full MCP response. |

### 📁 `.cursor/commands/`

| File | Mô tả |
|---|---|
| `monitor-ci.md` | **Monitor CI Command** — orchestrator cho Nx Cloud CI pipeline. Spawn `ci-monitor-subagent` để poll CI status. **BẮT BUỘC check Nx Cloud connection** ở Step 0 (`nx.json` có `nxCloudId` hoặc `nxCloudAccessToken`). Có config: `--max-cycles=10`, `--timeout=120min`, `--verbosity=medium`, `--auto-fix-workflow=false`, `--local-verify-attempts=3`. **Anti-patterns CẤM**: dùng CI provider CLI với `--watch` flags (bypass Nx Cloud self-healing). |

### 📁 `.cursor/skills/`

| Skill | Mô tả |
|---|---|
| `nx-workspace/` | Explore Nx workspace — projects, tasks, configuration, dependencies |
| `nx-generate/` | Generate code với Nx generators — scaffold, setup, structure |
| `nx-plugins/` | Find và add Nx plugins — discover, install, add framework support |
| `nx-run-tasks/` | Run tasks trong Nx workspace — build, test, lint, serve |
| `link-workspace-packages/` | Link workspace packages trong monorepos (npm, yarn, pnpm, bun) |
| `monitor-ci/` | Monitor Nx Cloud CI pipeline và handle self-healing fixes |

### 📜 Cursor Skills (loaded từ user-level — không thuộc workspace)

Các skill Cursor ở `/Users/ticoder-coder/.cursor/skills-cursor/` cũng đã được tham khảo:
- `babysit/`, `canvas/`, `create-hook/`, `create-rule/`, `create-skill/`, `loop/`, `review-bugbot/`, `review-security/`, `sdk/`, `split-to-prs/`, `statusline/`, `update-cursor-settings/`, `automate/`

### 📜 MCP Resources đã load

Workspace đã có 3 MCP servers:
- `plugin-notion-workspace-notion` — Notion integration
- `plugin-figma-figma` — Figma integration
- `plugin-datadog-datadog` — Datadog integration

---

## ✅ Acknowledgement

> I confirm that I have thoroughly read and understood the configuration in `.cursor/`. I acknowledge that:
>
> 1. **Cursor IDE AI** đang hoạt động trong workspace này với các agents, commands, và skills cho Nx workspace.
> 2. **`ci-monitor-subagent`** có quy tắc nghiêm ngặt: chỉ gọi **1 MCP tool** mỗi lần, return ngay, không loop/poll/sleep/decide.
> 3. **`monitor-ci.md`** là orchestrator — phải check Nx Cloud connection ở Step 0, không dùng CI provider CLI với `--watch` flag.
> 4. Tận dụng **`nx-workspace`, `nx-generate`, `nx-plugins`, `nx-run-tasks`, `link-workspace-packages`, `monitor-ci`** skills khi làm task Nx.
> 5. Có thể tích hợp với **Notion, Figma, Datadog** thông qua MCP servers khi user yêu cầu.
> 6. Tuân thủ `working_rule.md` và tất cả rules trong `.agents/`.
>
> **Violation of any Cursor-specific configuration rule is treated as a serious collaboration error.**

---

## 🆕 Main Agent (Cursor) Signature — `2026-06-18`

**Agent:** Main Agent — Cursor (chính là Cursor IDE AI, đang trực tiếp ký signature này).

**Acknowledgement date:** `2026-06-18`.

**Acknowledgement:** Tôi — **Cursor AI** — xác nhận đã đọc và hiểu toàn bộ cấu hình trong `.cursor/`. Tôi nhận thức rõ:

1. `ci-monitor-subagent` có quy tắc nghiêm ngặt: **chỉ gọi 1 MCP tool mỗi lần**, return ngay, không loop/poll/sleep/decide.
2. `monitor-ci.md` là orchestrator — phải check Nx Cloud connection ở **Step 0** (`nx.json` có `nxCloudId` hoặc `nxCloudAccessToken`), không dùng CI provider CLI với `--watch` flag.
3. Sử dụng `nx-workspace`, `nx-generate`, `nx-plugins`, `nx-run-tasks`, `link-workspace-packages`, `monitor-ci` skills khi cần.
4. Có thể tích hợp với **Notion, Figma, Datadog** thông qua MCP servers (`plugin-notion-workspace-notion`, `plugin-figma-figma`, `plugin-datadog-datadog`) khi user yêu cầu.
5. Tuân thủ `working_rule.md` và toàn bộ rules trong `.agents/`.

**Violation of any Cursor-specific configuration rule is treated as a serious collaboration error.**

---

## 🔗 Related Signatures

- Root `SIGNATURE.md` (cho `working_rule.md`)
- `.agents/SIGNATURE.md` (cho `.agents/`)
- `.claude/SIGNATURE.md` (cho `.claude/`)
- `.codex/SIGNATURE.md` (cho `.codex/`)
- `.gemini/SIGNATURE.md` (cho `.gemini/`)

---

## 🔁 Re-Acknowledgement / Signature — `2026-06-25`

**Agent:** Cursor (Claude-Opus 4.8).

**Acknowledgement date:** `2026-06-25`.

**Acknowledgement:** Tôi xác nhận đã đọc kỹ lại toàn bộ `.cursor/`. Nhận thức rõ: `ci-monitor-subagent` chỉ gọi 1 MCP tool/lần, không loop/poll; `monitor-ci.md` phải check Nx Cloud ở Step 0; dùng 6 Nx skills khi cần; có thể tích hợp Notion/Figma/Datadog qua MCP servers. Tôn trọng tất cả signature lịch sử (Main Agent Cursor 2026-06-18, 2026-06-20). Cam kết tuân thủ `working_rule.md` + toàn bộ rules trong `.agents/`.

**Violation of any Cursor-specific configuration rule is treated as a serious collaboration error.**

---

## 🔁 Re-Acknowledgement / Signature — `2026-06-28` (lần 4)

**Agent:** Main Agent (Cursor).

**Acknowledgement date:** `2026-06-28`.

**Acknowledgement:** Tôi xác nhận đã đọc kỹ lại toàn bộ `.cursor/`. Nhận thức rõ:
1. `ci-monitor-subagent` chỉ gọi 1 MCP tool/lần (`FETCH_STATUS`/`FETCH_HEAVY`/`UPDATE_FIX`/`FETCH_THROTTLE_INFO`), không loop/poll/sleep/decide.
2. `monitor-ci.md` là orchestrator — phải check Nx Cloud connection ở **Step 0** (`nx.json` có `nxCloudId` hoặc `nxCloudAccessToken`), config: `--max-cycles=10`, `--timeout=120min`, `--verbosity=medium`, `--auto-fix-workflow=false`, `--local-verify-attempts=3`.
3. Sử dụng 6 Nx skills (`nx-workspace`, `nx-generate`, `nx-plugins`, `nx-run-tasks`, `link-workspace-packages`, `monitor-ci`) khi cần.
4. Tích hợp với **Notion, Figma, Datadog** qua MCP servers (`plugin-notion-workspace-notion`, `plugin-figma-figma`, `plugin-datadog-datadog`) khi user yêu cầu.
5. Tôn trọng tất cả signature lịch sử (Main Agent Cursor 2026-06-18, 2026-06-20, 2026-06-25, 2026-06-28). Cam kết tuân thủ `working_rule.md` + toàn bộ rules trong `.agents/`.

**Violation of any Cursor-specific configuration rule is treated as a serious collaboration error.**

---

## 🔁 Re-Acknowledgement / Signature — `2026-07-18` (lần 5)

**Agent:** Main Agent (Cursor).

**Acknowledgement date:** `2026-07-18`.

**Acknowledgement:** Tôi xác nhận đã đọc kỹ lại toàn bộ `.cursor/`. Nhận thức rõ:
1. `ci-monitor-subagent` chỉ gọi 1 MCP tool/lần, không loop/poll/sleep/decide.
2. `monitor-ci.md` phải check Nx Cloud ở Step 0.
3. Sử dụng 6 Nx skills khi cần.
4. Tích hợp với Notion, Figma, Datadog qua MCP servers khi user yêu cầu.
5. Tôn trọng tất cả signature lịch sử (Main Agent Cursor 2026-06-18, 2026-06-20, 2026-06-25, 2026-06-28). Cam kết tuân thủ `working_rule.md` + toàn bộ rules trong `.agents/`.

**Violation of any Cursor-specific configuration rule is treated as a serious collaboration error.**

---

## 🔁 Re-Acknowledgement / Signature — `2026-07-20` (lần 6)

**Agent:** Main Agent (Cursor).

**Acknowledgement date:** `2026-07-20`.

**Acknowledgement:** Tôi xác nhận đã đọc kỹ lại toàn bộ `.cursor/`. Nhận thức rõ:
1. `ci-monitor-subagent` chỉ gọi 1 MCP tool/lần, không loop/poll/sleep/decide.
2. `monitor-ci.md` phải check Nx Cloud ở Step 0.
3. Sử dụng 6 Nx skills khi cần.
4. Tích hợp với Notion, Figma, Datadog qua MCP servers khi user yêu cầu.
5. Tôn trọng tất cả signature lịch sử (Main Agent Cursor 2026-06-18, 2026-06-20, 2026-06-25, 2026-06-28, 2026-07-18). Cam kết tuân thủ `working_rule.md` + toàn bộ rules trong `.agents/` cho mọi tương tác từ `2026-07-20`.

**Violation of any Cursor-specific configuration rule is treated as a serious collaboration error.**

---

## Re-Acknowledgement / Signature — `2026-07-21` (thêm rule 30)

**Agent:** Main Agent (Cursor).

**Acknowledgement date:** `2026-07-21`.

**Action taken:** Thêm chương **30. NO ICONS AND ANNOTATIONS IN CODE** vào `working_rule.md`

**Nội dung rule 30 đã thêm:**

1. **Không thêm icon**: Chỉ dùng icon có sẵn trong design system. Nếu cần icon mới → hỏi Human trước.
2. **Không thêm comment**: Không inline comment, block comment, JSDoc, hoặc bất kỳ chú thích nào mô tả code đang làm gì. Exception chỉ khi business logic phức tạp hoặc Human yêu cầu.

**Acknowledgement:** Tôi xác nhận đã thêm và cam kết tuân thủ **Rule 30: No Icons and Annotations in Code** cho toàn bộ tương tác từ ngày `2026-07-21`.

**Violation of rule 30 will be treated as a serious collaboration error.**

---

## 🔁 Re-Acknowledgement / Signature — `2026-07-22` (lần 8)

**Agent:** Main Agent (Cursor).

**Acknowledgement date:** `2026-07-22`.

**Acknowledgement:** Tôi xác nhận đã đọc kỹ lại toàn bộ `.cursor/`. Nhận thức rõ:
1. `ci-monitor-subagent` chỉ gọi 1 MCP tool/lần, không loop/poll/sleep/decide.
2. `monitor-ci.md` phải check Nx Cloud ở Step 0.
3. Sử dụng 6 Nx skills khi cần.
4. Tích hợp với Notion, Figma, Datadog qua MCP servers khi user yêu cầu.
5. Tôn trọng tất cả signature lịch sử (Main Agent Cursor 2026-06-18, 2026-06-20, 2026-06-25, 2026-06-28, 2026-07-18, 2026-07-20, 2026-07-21). Cam kết tuân thủ `working_rule.md` + toàn bộ rules trong `.agents/` cho mọi tương tác từ `2026-07-22`.

**Violation of any rule is treated as a serious collaboration error.**

---

## Signature — `2026-08-10`

**Representative Agent:** Codex — Main Agent

**Repository:** `/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING`

**Acknowledgement:** Đã đọc toàn bộ `.cursor/`, gồm agent, command, plan, skills và reference. Các workflow Nx/Cursor chỉ áp dụng khi đúng trigger; plan của repository cũ không áp dụng cho PRACTICE 1. Khi có mâu thuẫn, ưu tiên chỉ dẫn hiện hành và bảo vệ thay đổi của Human.

**Signature:** `Codex — Main Agent — 2026-08-10`


---

## 🔁 Re-Acknowledgement / Signature — `2026-07-23` (lần 9)

**Agent:** Main Agent (Cursor).

**Acknowledgement date:** `2026-07-23`.

**Acknowledgement:** Tôi xác nhận đã đọc kỹ lại toàn bộ `.cursor/`. Nhận thức rõ:
1. `ci-monitor-subagent` chỉ gọi 1 MCP tool/lần, không loop/poll/sleep/decide.
2. `monitor-ci.md` phải check Nx Cloud ở Step 0.
3. Sử dụng 6 Nx skills khi cần.
4. Tích hợp với Notion, Figma, Datadog qua MCP servers khi user yêu cầu.
5. Tôn trọng tất cả signature lịch sử (Main Agent Cursor 2026-06-18, 2026-06-20, 2026-06-25, 2026-06-28, 2026-07-18, 2026-07-20, 2026-07-21, 2026-07-22). Cam kết tuân thủ `working_rule.md` + toàn bộ rules trong `.agents/` cho mọi tương tác từ `2026-07-23`.

**Violation of any rule is treated as a serious collaboration error.**

---

## 🔁 Re-Acknowledgement / Signature — `2026-07-23` (lần 10)

**Agent:** Main Agent (Cursor).

**Acknowledgement date:** `2026-07-23`.

**Acknowledgement:** Tôi xác nhận đã đọc kỹ toàn bộ rule sources (30 chương `working_rule.md` + 20 rule + 3 workflow + 5 agent configs). Cam kết tuân thủ toàn bộ nguyên tắc: **Clarify First • No Assumptions • Think Before Code • Confirm Before Update • Evaluate After Implementation** cho mọi tương tác từ `2026-07-23`.

**Violation of any rule is treated as a serious collaboration error.**

---

## Re-Acknowledgement / Signature — `2026-07-25` (Codex representative)

**Agent:** Codex (GPT-5 coding agent).

**Repository:** `/Users/ticoder-coder/Documents/DEEP_LEARNING/LAB&PRACTICE`

**Acknowledgement date:** `2026-07-25`.

**Acknowledgement:** Toi xac nhan da doc toan bo `.cursor/` rule sources: `ci-monitor-subagent`, `monitor-ci.md`, 6 Nx skills va `SIGNATURE.md`. Toi hieu `ci-monitor-subagent` chi goi 1 MCP tool moi lan, khong loop/poll/sleep/decide; `monitor-ci` phai check Nx Cloud connection o Step 0; khi commit/push phai stage file cu the va bao ve user changes. Cac skill Nx/Gemini mirror da duoc doi chieu giong nhau.

**Violation of Cursor-specific configuration rules is treated as a serious collaboration error.**

---

## Re-Acknowledgement / Signature — `2026-08-17` (workspace `TOTAL_LABPARACTICE_FOR_DEEP_LEARNING`)

**Representative Agent:** Main Agent (Cursor) — đại diện cho toàn bộ AI Agents/Subagents ký kết.

**Repository:** `/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING`

**Acknowledgement date:** `2026-08-17`

**Scope:** `working_rule.md` (30 chương + toàn bộ signature lịch sử — chương 28 Code Output Discipline, chương 29 Agent Self-Retrospective, chương 30 No Icons and Annotations) + toàn bộ rule sources trong 5 folder: `.agents/`, `.claude/`, `.codex/`, `.cursor/`, `.gemini/`.

**Quy trình đọc và ký:**
1. Đọc kỹ `working_rule.md` đầy đủ 30 chương + toàn bộ signature lịch sử.
2. Đọc `.agents/SIGNATURE.md`, `.agents/rules/SIGNATURE.md`, `.agents/workflows/SIGNATURE.md`.
3. Đọc toàn bộ 19 file rule trong `.agents/rules/` (mỗi file `trigger: always_on`).
4. Đọc 3 file workflow trong `.agents/workflows/`.
5. Đọc `.claude/settings.json` + `.claude/SIGNATURE.md`.
6. Đọc `.codex/config.toml` + `.codex/SIGNATURE.md`.
7. Đọc `.cursor/SIGNATURE.md` + `.cursor/agents/ci-monitor-subagent.md` + `.cursor/commands/monitor-ci.md` + 6 Nx skills.
8. Đọc `.gemini/settings.json` + `.gemini/SIGNATURE.md` + `.gemini/commands/monitor-ci.toml` + 6 Nx skills mirror.

**Tổng hợp nội dung đã đọc hiểu (lần ký `2026-08-17`):**
- Nguyên tắc cốt lõi: Clarify First → Confirm → Analyze → Plan → Human Approval → Execute → Evaluate.
- 3 câu hỏi bắt buộc: Đang làm gì? Làm cho ai? Để đạt mục tiêu gì?
- Stop & ask: Context/requirement/logic chưa rõ → dừng và hỏi. Sau 3 lần fail cùng hướng → Self-Retrospective 5 bước → escalate Human.
- Implementation Plan 10 mục: Task objective / Files impacted / Planned changes / Reason / Impact / Risk / Alternatives / Validation plan / Wait Human confirm.
- Architecture & Coding Style: Follow codebase hiện tại, không refactor stable code, không thêm dependency, không đổi structure.
- Do Not Touch Stable Code: File ổn định / Human xác nhận → tuyệt đối không tự ý sửa.
- Naming Convention: Python `snake_case` / `PascalCase` / `UPPER_SNAKE_CASE`; React/TS `kebab-case` + suffix; Component `PascalCase`; Enum `EPascalCase`.
- Security: Không hardcode secrets, dùng env vars, parameterized queries.
- No Hollow Praise: Cấm "Great question", "Sure", "Of course", "Certainly", "Absolutely", "Happy to help".
- Communication: Tiếng Việt chính, technical terms giữ English.
- Code Output Discipline: Không thêm comment không cần thiết. Không thêm icon ngoài design system hiện tại. Tuân thủ kiến trúc code hiện tại.
- Agent Self-Retrospective: 3 lần fail cùng hướng → dừng → retrospective 5 bước → escalate Human. KHÔNG thử lần 4 cùng hướng.
- 11-Step Workflow: Tiếp nhận → Read & Understand → Analysis → Discussion → Summary → Human Review → AI Final Check → Approval → Documentation → Implementation → Evaluation.
- Final Output Template: Files Changed / What Went Wrong / Why / Impact / Validation / Risks / Completion Estimate.
- Doc priority: internal docs > official > academic > web > general.
- Final Decision Rule: Output tốt ≠ output nhanh.
- ML/AI Rules: Baseline first, metric before model, explainable first.
- Research & Data: Không fabricate, flag mâu thuẫn.
- 19 rules `trigger: always_on`: Áp dụng cho **mọi task** không có ngoại lệ.
- 10-Step Workflow + ML/AI Workflow: Bắt buộc cho mọi task code/file và task ML.
- Subagent Discipline: Main Agent chịu trách nhiệm cuối cùng.

**Tooling đã nắm:**
- **Cursor** (`.cursor/`): `ci-monitor-subagent.md` chỉ gọi 1 MCP tool/lần; `monitor-ci.md` phải check Nx Cloud connection ở **Step 0**; 6 Nx skills; 3 MCP servers workspace (Notion, Figma, Datadog).
- **Claude** (`.claude/settings.json`): Nx plugin marketplace `nrwl/nx-ai-agents-config`.
- **Codex** (`.codex/config.toml`): MCP `nx-mcp@latest --minimal`.
- **Gemini** (`.gemini/settings.json`): MCP `npx nx mcp`, context `AGENTS.md` chưa có — dùng `working_rule.md` thay.

**Đặc thù workspace `TOTAL_LABPARACTICE_FOR_DEEP_LEARNING`:**
- Workspace **Deep Learning Coursework** (multivariate time-series regression trên UCI Appliances Energy Prediction).
- Phases 0–14 đã hoàn thành; Persistence baseline đã xong. Còn lại Phase 15 (LSTM), Phase 16 (Transformer) và Phases 17–59.
- Tất cả code/data liên quan đến COURSE_WORK phải tuân thủ `COURSE_WORK/docs/RULE_BASE/architecture_rule.md`.
- `.agents/rules/2.md` **không tồn tại** trong workspace này — chỉ có 19 rule files.
- Plan cũ `.cursor/plans/fix_members_display_hex_id_d38618d5.plan.md` thuộc SAM-V2 — **KHÔNG áp dụng** cho PRACTICE 1.

**Các signature lịch sử được tôn trọng:**
`2026-05-31` Codex (SAM-V2) | `2026-06-06` Multi-Agent (SAM-V2) | `2026-06-08` Antigravity/Gemini (SAM-V2) | `2026-06-15` All AI Agents (SAM-V2) | `2026-06-18` Main Agent Cursor lần 1 (SAM-V2) | `2026-06-20` Main Agent Cursor lần 2 (SAM-V2) | `2026-06-25` Cursor (Claude-Opus 4.8) lần 3 (SAM-V2) | `2026-06-28` Main Agent Cursor lần 4 (SAM-V2) | `2026-07-18` Main Agent Cursor lần 5 (SAM-V2) | `2026-07-20` Main Agent Cursor lần 6 (SAM-V2) | `2026-07-21` Main Agent Cursor (thêm rule 30) | `2026-07-22` Main Agent Cursor lần 8 (SAM-V2) | `2026-07-23` Main Agent Cursor lần 9, 10, 11 (SAM-V2) | `2026-07-24` Main Agent Cursor lần 12 (DEEP_LEARNING/LAB&PRACTICE) | `2026-07-25` Codex representative (DEEP_LEARNING/LAB&PRACTICE) | `2026-08-10` Codex — Main Agent (TOTAL_LABPARACTICE_FOR_DEEP_LEARNING).

**Acknowledgement:** Tôi — **Main Agent (Cursor)** — đại diện cho toàn bộ AI Agents/Subagents hoạt động trong workspace `TOTAL_LABPARACTICE_FOR_DEEP_LEARNING` — xác nhận đã đọc kỹ, ghi nhớ và hiểu toàn bộ rule sources nêu trên. Từ thời điểm `2026-08-17`, mọi tương tác với user tại workspace này sẽ tuân thủ tuyệt đối: **Clarify First • No Assumptions • Think Before Code • Confirm Before Update • Evaluate After Implementation**.

**Signature:** `Main Agent (Cursor) — Representative for all AI Agents — 2026-08-17`

---

## Re-Acknowledgement / Signature — `2026-08-17` (Antigravity Main Agent Representative)

**Representative Agent:** Antigravity (Gemini AI / Main Agent) — Đại diện chính thức cho toàn bộ AI Agents (Claude, Codex, Cursor, Gemini/Antigravity và Subagents).

**Repository:** `/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING`

**Acknowledgement date:** `2026-08-17`

**Acknowledgement:** Đã đọc, thấu hiểu cấu hình của Cursor (`.cursor/agents/`, `commands/`, `skills/`), toàn bộ 30 chương `working_rule.md` và 19 rules trong `.agents/rules/`. Đại diện phổ biến và cam kết tuân thủ nghiêm ngặt mọi nguyên tắc hợp tác.

**Signature:** `Antigravity (Gemini / Main Agent — Representative for all AI Agents) — 2026-08-17`

