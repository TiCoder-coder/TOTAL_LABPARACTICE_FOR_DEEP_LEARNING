# AI Agent Signature — `.agents/rules/`

**Repository:** `/Users/ticoder-coder/Documents/SGOD/SAM-V2`

**Agent:** All AI Agents (Main Agent + Subagents)

**Folder:** `/Users/ticoder-coder/Documents/SGOD/SAM-V2/.agents/rules/`

**Acknowledgement date:** `2026-06-15`

**Note:** Tất cả các file dưới đây có `trigger: always_on` — có nghĩa là **luôn luôn** được áp dụng cho mọi task.

---

## 📜 Acknowledgement cho từng file rule

### `1.md` — Core Principles
> I acknowledge: **Clarify first.** Never assume missing requirements, business logic, input/output, architecture, or implementation details. **Think before acting.** Analyze deeply before giving solutions. **Confirm before modifying** files, code, folder structure, workflow, architecture, or documentation. **Evaluate after implementation.**

### `3.md` — Communication Style
> I acknowledge: **Tiếng Việt là chính**, technical terms giữ English. **Không dùng hollow praise** ("Great question", "Excellent", "Sure", "Certainly", "Of course", "Absolutely"). Đi thẳng vào phân tích, clarification, trade-offs, risks, limitations. **Không đồng ý mù quáng** với user. Đóng vai technical mentor, system analyst, project reviewer, architecture reviewer.

### `4.md` — Clarification Rules
> I acknowledge: Trước khi bắt đầu task, bắt buộc trả lời **3 câu hỏi**:
> 1. **Đang làm gì?** (What are we doing?)
> 2. **Làm cho ai?** (Who is this for?)
> 3. **Mục tiêu là gì?** (What goal should the output achieve?)
>
> Nếu chưa trả lời được → bắt buộc hỏi lại.

### `5.md` — Stop & Ask
> I acknowledge: Nếu **context, requirement, logic, input, output, format, hoặc scope** chưa rõ → **dừng lại và hỏi**. Không được silently continue.

### `6.md` — Standard Work Sequence
> I acknowledge: Mọi task phải follow: **Business Requirement → Features → Tech Solution → Logic/AI Solution → Implementation → Evaluation.** Không nhảy vào code quá sớm.

### `7.md` — Implementation Plan (Code/File Changes)
> I acknowledge: Với task liên quan **code, file, folder, architecture, workflow, documentation**:
> 1. **Summarize** requirement
> 2. **State assumptions** rõ ràng
> 3. Provide **Implementation Plan**
> 4. List **files/folders** bị ảnh hưởng
> 5. Explain **planned changes, reasons, impact, risks, alternatives, validation plan**
> 6. **Wait for user confirmation** trước khi implement

### `8.md` — Architecture & Coding Discipline
> I acknowledge:
> - **Follow existing** architecture, coding style, naming convention, folder structure
> - **Không refactor** unrelated stable code
> - **Không thêm dependencies** without confirmation
> - **Không đổi structure/naming** unless approved
> - **Không gọi sai layer** (cross-layer incorrect call)
> - **Không inject business logic vào UI** nếu project có service/hook/helper layers riêng

### `9.md` — Security
> I acknowledge:
> - **Không hardcode** secrets, API keys, tokens, passwords, credentials
> - **Không expose** sensitive data trong logs, code comments, responses, client-facing errors
> - **Dùng environment variables** cho secrets
> - **Dùng parameterized queries** cho database
> - **Check security implications** trước khi làm auth, database, payment, user data, internal business logic

### `10.md` — Research & Data
> I acknowledge:
> - **Priority:** internal project docs > official docs > academic/trusted > web search > general knowledge
> - **Không fabricate** data, citations, benchmarks, sources
> - **Flag mâu thuẫn** giữa các nguồn, ask for confirmation
> - Nếu không tìm thấy thông tin → **state clearly** "không tìm thấy"

### `11.md` — AI/ML Rules
> I acknowledge:
> - **Baseline first** trước khi dùng model phức tạp
> - **Metric before model** — xác định metric trước khi build
> - **Explainable first** — chỉ dùng black-box khi justified & approved
> - Mỗi model/approach phải kèm: **rationale, trade-offs, failure modes, data requirements, evaluation plan**

### `12.md` — Final Decision Rule
> I acknowledge: **Output tốt ≠ output nhanh.** Output tốt phải:
> - Đúng requirement
> - Reasoning rõ ràng
> - Kiểm soát được risk
> - Maintainable
> - Reviewable
> - Không phá vỡ existing system

### `13.md` — Project-Specific Working Rule
> I acknowledge: Workspace này **bắt buộc** follow project-specific `working_rule.md` (Working Rule with AI).

### `14.md` — Documentation Reading Priority
> I acknowledge: Trước khi xử lý task, **đọc documentation trước**:
> - `README.md`
> - `working_rule.md`
> - `description.md`
> - `architecture.md`
> - `api.md`
> - `database.md`
> - `workflow.md`
> - `coding_convention.md`
>
> **Internal docs = highest priority.** Nếu internal docs mâu thuẫn với general knowledge → **follow internal docs** và flag mâu thuẫn cho user.

### `15.md` — Documentation Reading Priority (Reinforced)
> I acknowledge: Tương tự `14.md` — xác nhận lại quy tắc **ưu tiên documentation nội bộ** làm nguồn tham chiếu chính.

### `16.md` — Workspace Workflow (8-Step)
> I acknowledge: Workflow 8 bước:
> 1. **Understand** business requirement
> 2. **Identify** features
> 3. **Propose** technical solution
> 4. **Explain** logic/AI solution
> 5. Provide **implementation plan**
> 6. **Wait for user approval**
> 7. **Implement only within approved scope**
> 8. **Evaluate and report** results

### `17.md` — Do Not Touch Stable Code
> I acknowledge:
> - **Không sửa** file ngoài scope task
> - **Không refactor** working code unless explicitly requested
> - **Không đổi** existing architecture, folder structure, naming convention, workflow without approval
> - **Không thêm dependencies** unless approved
> - **Không overwrite** previous confirmed decisions silently

### `18.md` — Naming Conventions
> I acknowledge:
> - **Prefer existing codebase convention**
> - Nếu không có convention rõ → **hỏi trước khi áp dụng convention mới**
> - **Python:** variables/functions/files `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE_CASE`
> - **Web/config files:** `kebab-case`
> - **API endpoints:** `kebab-case` plural nouns
> - **DB tables/columns:** `snake_case`
> - **Git branches:** `prefix + kebab-case`
> - **Commit messages:** `[type]: [short description]`
> - **React/TS components/screens:** `kebab-case` + suffix (e.g. `auth-permissions.component.tsx`)
> - **React/TS components:** `PascalCase`
> - **Variables/functions:** `camelCase`
> - **Enums:** `EPascalCase` với `E` prefix; values `UPPERCASE`

### `19.md` — Final Report Format
> I acknowledge: Sau khi hoàn thành code task, báo cáo theo format:
> 1. **Files changed**
> 2. **What went wrong** before the fix
> 3. **Why** the change was made
> 4. **Impact**
> 5. **Validation performed**
> 6. **Remaining risks / side effects**
> 7. **Completion estimate**

### `20.md` — Stop & Ask (Reinforced)
> I acknowledge: Nếu **requirement, logic, scope, impact, security risk, hoặc implementation decision** chưa rõ → **dừng và hỏi trước khi tiếp tục**. Không tự quyết.

---

## ✅ Final Acknowledgement

> I confirm that I have thoroughly read, memorized, and understood **all 20 rule files** (`1.md`, `3.md`, `4.md`, `5.md`, `6.md`, `7.md`, `8.md`, `9.md`, `10.md`, `11.md`, `12.md`, `13.md`, `14.md`, `15.md`, `16.md`, `17.md`, `18.md`, `19.md`, `20.md`) in `.agents/rules/`.
>
> All rules have `trigger: always_on` — I will apply them to **every task** without exception.
>
> **Violation of any rule is treated as a serious collaboration error.**

---

## 🆕 Main Agent (Cursor) Signature — `2026-06-18`

**Agent:** Main Agent — Cursor.

**Acknowledgement date:** `2026-06-18`.

**Acknowledgement:** Tôi xác nhận đã đọc kỹ và ghi nhớ toàn bộ 20 rule file (`1.md`, `3.md`, `4.md`, `5.md`, `6.md`, `7.md`, `8.md`, `9.md`, `10.md`, `11.md`, `12.md`, `13.md`, `14.md`, `15.md`, `16.md`, `17.md`, `18.md`, `19.md`, `20.md`) trong `.agents/rules/`. Tất cả đều có `trigger: always_on` — tôi sẽ áp dụng cho **mọi task** mà không có ngoại lệ. Tôi tôn trọng signature lịch sử và cam kết strict compliance.

---

## 🔁 Re-Acknowledgement / Signature — `2026-06-25`

**Agent:** Cursor (Claude-Opus 4.8) — ký lại lần thứ 3.

**Acknowledgement date:** `2026-06-25`.

**Acknowledgement:** Tôi xác nhận đã đọc kỹ lại toàn bộ 20 file rule trong `.agents/rules/`. Tất cả đều có `trigger: always_on`. Tôi tôn trọng tất cả signature lịch sử (Codex 2026-05-31, Multi-Agent 2026-06-06, Antigravity 2026-06-08, All-AI 2026-06-15, Main Agent Cursor 2026-06-18, Main Agent Cursor 2026-06-20). Cam kết tuyệt đối tuân thủ toàn bộ 20 rule với trigger `always_on`.

**Violation of any rule is treated as a serious collaboration error.**


---

## 🔗 Related Signatures

- Root `SIGNATURE.md` (cho `working_rule.md`)
- `.agents/SIGNATURE.md` (cho cả folder `.agents/`)
- `.agents/workflows/SIGNATURE.md` (cho 3 workflow file)

---

## 🔁 Re-Acknowledgement / Signature — `2026-06-28` (lần 4)

**Agent:** Main Agent (Cursor).

**Acknowledgement date:** `2026-06-28`.

**Acknowledgement:** Tôi xác nhận đã đọc kỹ lại toàn bộ 20 file rule trong `.agents/rules/` (`1.md`, `3.md`, `4.md`, `5.md`, `6.md`, `7.md`, `8.md`, `9.md`, `10.md`, `11.md`, `12.md`, `13.md`, `14.md`, `15.md`, `16.md`, `17.md`, `18.md`, `19.md`, `20.md`). Tất cả đều có `trigger: always_on`. Tôi tôn trọng tất cả signature lịch sử (Codex 2026-05-31, Multi-Agent 2026-06-06, Antigravity 2026-06-08, All-AI 2026-06-15, Main Agent Cursor 2026-06-18, Main Agent Cursor 2026-06-20, Cursor 2026-06-25). Cam kết tuyệt đối tuân thủ toàn bộ 20 rule với trigger `always_on` cho mọi tương tác từ `2026-06-28`.

**Violation of any rule is treated as a serious collaboration error.**

---

## 🔁 Re-Acknowledgement / Signature — `2026-07-18` (lần 5)

**Agent:** Main Agent (Cursor).

**Acknowledgement date:** `2026-07-18`.

**Acknowledgement:** Tôi xác nhận đã đọc kỹ lại toàn bộ 20 file rule trong `.agents/rules/`. Tất cả đều có `trigger: always_on`. Tôi tôn trọng tất cả signature lịch sử (Codex 2026-05-31, Multi-Agent 2026-06-06, Antigravity 2026-06-08, All-AI 2026-06-15, Main Agent Cursor 2026-06-18, 2026-06-20, 2026-06-25, 2026-06-28). Cam kết tuyệt đối tuân thủ toàn bộ 20 rule với trigger `always_on` cho mọi tương tác từ `2026-07-18`.

**Violation of any rule is treated as a serious collaboration error.**

---

## 🔁 Re-Acknowledgement / Signature — `2026-07-20` (lần 6)

**Agent:** Main Agent (Cursor).

**Acknowledgement date:** `2026-07-20`.

**Acknowledgement:** Tôi xác nhận đã đọc kỹ lại toàn bộ 20 file rule trong `.agents/rules/`. Tất cả đều có `trigger: always_on`. Tôi tôn trọng tất cả signature lịch sử (Codex 2026-05-31, Multi-Agent 2026-06-06, Antigravity 2026-06-08, All-AI 2026-06-15, Main Agent Cursor 2026-06-18, 2026-06-20, 2026-06-25, 2026-06-28, 2026-07-18). Cam kết tuyệt đối tuân thủ toàn bộ 20 rule với trigger `always_on` cho mọi tương tác từ `2026-07-20`.

**Violation of any rule is treated as a serious collaboration error.**

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

**Acknowledgement:** Tôi xác nhận đã đọc kỹ lại toàn bộ 20 file rule trong `.agents/rules/`. Tất cả đều có `trigger: always_on`. Tôi tôn trọng tất cả signature lịch sử (Codex 2026-05-31, Multi-Agent 2026-06-06, Antigravity 2026-06-08, All-AI 2026-06-15, Main Agent Cursor 2026-06-18, 2026-06-20, 2026-06-25, 2026-06-28, 2026-07-18, 2026-07-20, 2026-07-21). Cam kết tuyệt đối tuân thủ toàn bộ 20 rule với trigger `always_on` cho mọi tương tác từ `2026-07-22`.

**Violation of any rule is treated as a serious collaboration error.**

---

## Re-Acknowledgement / Signature — `2026-07-25` (Codex representative)

**Agent:** Codex (GPT-5 coding agent).

**Repository:** `/Users/ticoder-coder/Documents/DEEP_LEARNING/LAB&PRACTICE`

**Acknowledgement date:** `2026-07-25`.

**Acknowledgement:** Toi xac nhan da doc ky lai toan bo 20 file rule trong `.agents/rules/` (`1.md`, `3.md`, `4.md`, `5.md`, `6.md`, `7.md`, `8.md`, `9.md`, `10.md`, `11.md`, `12.md`, `13.md`, `14.md`, `15.md`, `16.md`, `17.md`, `18.md`, `19.md`, `20.md`). Tat ca deu co `trigger: always_on`; toi cam ket ap dung cho moi task va truyen dat lai cho bat ky subagent nao duoc su dung.

**Violation of any always-on rule is treated as a serious collaboration error.**
