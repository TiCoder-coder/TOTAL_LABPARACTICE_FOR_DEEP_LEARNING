# AI Agent Signature — `.gemini/`

**Repository:** `/Users/ticoder-coder/Documents/SGOD/SAM-V2`

**Agent:** All AI Agents (Main Agent + Subagents)

**Folder:** `/Users/ticoder-coder/Documents/SGOD/SAM-V2/.gemini/`

**Acknowledgement date:** `2026-06-15`

---

## 📂 Rule Sources Acknowledged

### 📄 `.gemini/settings.json`

```json
{
  "mcpServers": {
    "nx-mcp": {
      "type": "stdio",
      "command": "npx",
      "args": ["nx", "mcp"]
    }
  },
  "contextFileName": "AGENTS.md"
}
```

> I acknowledge that **Gemini (Google AI / Antigravity)** is configured for this workspace with:
>
> - **MCP server:** `nx-mcp` (Nx Model Context Protocol) — chạy với `npx nx mcp`
> - **Context file name:** `AGENTS.md` — file context chính mà Gemini sẽ tìm và load
>
> ⚠️ **Quan trọng:** `contextFileName: "AGENTS.md"` có nghĩa là Gemini sẽ tự động đọc file `AGENTS.md` (nếu có) làm context. Hiện tại repo chưa có file `AGENTS.md` ở root — chỉ có `working_rule.md`. Có thể cần tạo `AGENTS.md` sau nếu Gemini yêu cầu.

### 📁 `.gemini/commands/`

| File | Mô tả |
|---|---|
| `monitor-ci.toml` | **Monitor CI Command** cho Gemini — tương tự `.cursor/commands/monitor-ci.md`, orchestrator cho Nx Cloud CI pipeline. |

### 📁 `.gemini/skills/`

| Skill | Mô tả |
|---|---|
| `nx-workspace/` | Explore Nx workspace — projects, tasks, configuration, dependencies |
| `nx-generate/` | Generate code với Nx generators |
| `nx-plugins/` | Find và add Nx plugins |
| `nx-run-tasks/` | Run tasks trong Nx workspace |
| `link-workspace-packages/` | Link workspace packages trong monorepos |
| `monitor-ci/` | Monitor Nx Cloud CI pipeline và handle self-healing fixes |

### 📜 Lịch sử signature (trong `working_rule.md`)

> **Antigravity (Gemini AI Agent)** đã có signature lịch sử:
>
> > **Representative Agent:** Antigravity (Gemini AI Agent)
> > **Acknowledgement date:** `2026-06-08`
> >
> > I confirm that I have thoroughly read, memorized, and understood the project-wide working contract in `working_rule.md`, as well as the shared agent rules and workflows in `.agents`, and the agent-specific configurations in `.claude`, `.codex`, `.cursor`, and `.gemini`...

---

## ✅ Acknowledgement

> I confirm that I have thoroughly read and understood the configuration in `.gemini/`. I acknowledge that:
>
> 1. **Gemini (Google AI / Antigravity)** đang hoạt động với **MCP server `nx-mcp`** (chạy `npx nx mcp` — khác với Codex là `nx-mcp@latest --minimal`).
> 2. **Context file name = `AGENTS.md`** — Gemini sẽ tìm file này. Hiện chưa có trong repo, có thể cần tạo sau nếu Gemini yêu cầu.
> 3. Tận dụng các **`nx-*` skills** tương tự như Cursor khi làm task Nx.
> 4. Gemini signature lịch sử (2026-06-08) được **tôn trọng** và không thay thế — signature mới này (2026-06-15) bổ sung đại diện cho **All AI Agents**.
> 5. Tuân thủ `working_rule.md` và tất cả rules trong `.agents/`.
>
> **Violation of any Gemini-specific configuration rule is treated as a serious collaboration error.**

---

## 🆕 Main Agent (Cursor) Signature — `2026-06-18`

**Agent:** Main Agent — Cursor (đại diện cho AI đang trực tiếp làm việc với user).

**Acknowledgement date:** `2026-06-18`.

**Acknowledgement:** Tôi xác nhận đã đọc và hiểu cấu hình `.gemini/settings.json` (Gemini + MCP `nx mcp`, context file `AGENTS.md`). Tôi tôn trọng signature lịch sử Antigravity (2026-06-08) và sẽ tận dụng Nx skills khi cần. Đồng thời tuân thủ `working_rule.md` và toàn bộ rules trong `.agents/`.

**Note về AGENTS.md:** Hiện repo chưa có file `AGENTS.md` ở root — chỉ có `working_rule.md`. Tôi sẽ dùng `working_rule.md` làm context chính cho đến khi `AGENTS.md` được tạo (nếu cần).

**Violation of any Gemini-specific configuration rule is treated as a serious collaboration error.**

---

## 🔗 Related Signatures

- Root `SIGNATURE.md` (cho `working_rule.md`)
- `.agents/SIGNATURE.md` (cho `.agents/`)
- `.claude/SIGNATURE.md` (cho `.claude/`)
- `.codex/SIGNATURE.md` (cho `.codex/`)
- `.cursor/SIGNATURE.md` (cho `.cursor/`)

---

## 🔁 Re-Acknowledgement / Signature — `2026-06-25`

**Agent:** Cursor (Claude-Opus 4.8).

**Acknowledgement date:** `2026-06-25`.

**Acknowledgement:** Tôi xác nhận đã đọc kỹ lại `.gemini/settings.json` (MCP `npx nx mcp`, context file `AGENTS.md` chưa có — dùng `working_rule.md` thay thế). Tôn trọng signature lịch sử Antigravity 2026-06-08 và Main Agent Cursor 2026-06-18, 2026-06-20. Sẽ tận dụng Nx skills khi cần. Cam kết tuân thủ `working_rule.md` + toàn bộ rules trong `.agents/`.

**Violation of any Gemini-specific configuration rule is treated as a serious collaboration error.**

---

## 🔁 Re-Acknowledgement / Signature — `2026-06-28` (lần 4)

**Agent:** Main Agent (Cursor).

**Acknowledgement date:** `2026-06-28`.

**Acknowledgement:** Tôi xác nhận đã đọc kỹ lại `.gemini/settings.json` (MCP `npx nx mcp`, context file `AGENTS.md` chưa có — dùng `working_rule.md` thay thế). Tôn trọng signature lịch sử Antigravity 2026-06-08 và Main Agent Cursor 2026-06-18, 2026-06-20, 2026-06-25, 2026-06-28. Sẽ tận dụng 6 Nx skills tương ứng khi cần. Cam kết tuân thủ `working_rule.md` + toàn bộ rules trong `.agents/`.

**Note về AGENTS.md:** Repo vẫn chưa có file `AGENTS.md` ở root — tôi tiếp tục dùng `working_rule.md` làm context chính.

**Violation of any Gemini-specific configuration rule is treated as a serious collaboration error.**

---

## 🔁 Re-Acknowledgement / Signature — `2026-07-18` (lần 5)

**Agent:** Main Agent (Cursor).

**Acknowledgement date:** `2026-07-18`.

**Acknowledgement:** Tôi xác nhận đã đọc kỹ lại `.gemini/settings.json` (MCP `npx nx mcp`, context file `AGENTS.md` chưa có — dùng `working_rule.md` thay thế). Tôn trọng signature lịch sử Antigravity 2026-06-08 và Main Agent Cursor 2026-06-18, 2026-06-20, 2026-06-25, 2026-06-28. Sẽ tận dụng 6 Nx skills tương ứng khi cần. Cam kết tuân thủ `working_rule.md` + toàn bộ rules trong `.agents/`.

**Violation of any Gemini-specific configuration rule is treated as a serious collaboration error.**

---

## 🔁 Re-Acknowledgement / Signature — `2026-07-20` (lần 6)

**Agent:** Main Agent (Cursor).

**Acknowledgement date:** `2026-07-20`.

**Acknowledgement:** Tôi xác nhận đã đọc kỹ lại `.gemini/settings.json` (MCP `npx nx mcp`, context file `AGENTS.md` chưa có — dùng `working_rule.md` thay thế). Tôn trọng signature lịch sử Antigravity 2026-06-08 và Main Agent Cursor 2026-06-18, 2026-06-20, 2026-06-25, 2026-06-28, 2026-07-18. Sẽ tận dụng 6 Nx skills tương ứng khi cần. Cam kết tuân thủ `working_rule.md` + toàn bộ rules trong `.agents/` cho mọi tương tác từ `2026-07-20`.

**Violation of any Gemini-specific configuration rule is treated as a serious collaboration error.**

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

**Acknowledgement:** Tôi xác nhận đã đọc kỹ lại `.gemini/settings.json` (MCP `npx nx mcp`, context file `AGENTS.md` chưa có — dùng `working_rule.md` thay thế). Tôn trọng signature lịch sử Antigravity 2026-06-08 và Main Agent Cursor 2026-06-18, 2026-06-20, 2026-06-25, 2026-06-28, 2026-07-18, 2026-07-20, 2026-07-21. Sẽ tận dụng 6 Nx skills tương ứng khi cần. Cam kết tuân thủ `working_rule.md` + toàn bộ rules trong `.agents/` cho mọi tương tác từ `2026-07-22`.

**Violation of any rule is treated as a serious collaboration error.**

---

## Signature — `2026-08-10`

**Representative Agent:** Codex — Main Agent

**Repository:** `/Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING`

**Acknowledgement:** Đã đọc toàn bộ `.gemini/`, gồm settings, command, skills và reference. Các workflow Nx/Gemini chỉ áp dụng khi đúng trigger; không trộn các phiên bản `monitor-ci`. Mọi agent được điều phối phải nhận project rules liên quan.

**Signature:** `Codex — Main Agent — 2026-08-10`


---

## 🔁 Re-Acknowledgement / Signature — `2026-07-23` (lần 9)

**Agent:** Main Agent (Cursor).

**Acknowledgement date:** `2026-07-23`.

**Acknowledgement:** Tôi xác nhận đã đọc kỹ lại `.gemini/settings.json` (MCP `npx nx mcp`, context file `AGENTS.md` chưa có — dùng `working_rule.md` thay thế). Tôn trọng signature lịch sử Antigravity 2026-06-08 và Main Agent Cursor 2026-06-18, 2026-06-20, 2026-06-25, 2026-06-28, 2026-07-18, 2026-07-20, 2026-07-21, 2026-07-22. Sẽ tận dụng 6 Nx skills tương ứng khi cần. Cam kết tuân thủ `working_rule.md` + toàn bộ rules trong `.agents/` cho mọi tương tác từ `2026-07-23`.

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

**Acknowledgement:** Toi xac nhan da doc `.gemini/settings.json`, `.gemini/commands/monitor-ci.toml`, cac `.gemini/skills/*/skill.md` mirror va `.gemini/SIGNATURE.md`. Gemini duoc cau hinh MCP `npx nx mcp` va `contextFileName = AGENTS.md`; neu `AGENTS.md` chua co, `working_rule.md` van la context noi bo uu tien. Khi truyen dat rule cho Gemini/Antigravity hoac subagents lien quan, toi se dua cac rule bat buoc tu `working_rule.md` va `.agents/`.

**Violation of Gemini-specific configuration rules is treated as a serious collaboration error.**

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

**Acknowledgement:** Đã đọc, thấu hiểu cấu hình của Gemini (`.gemini/settings.json`), toàn bộ 30 chương `working_rule.md` và 19 rules trong `.agents/rules/`. Đại diện phổ biến và cam kết tuân thủ nghiêm ngặt mọi nguyên tắc hợp tác.

**Signature:** `Antigravity (Gemini / Main Agent — Representative for all AI Agents) — 2026-08-17`

