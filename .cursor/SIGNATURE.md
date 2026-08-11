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
