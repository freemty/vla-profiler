# Changelog

## v0.2.1 — 2026-04-11

### 修复
- Dashboard: denoising chart inline styles 替换为 CSS classes
- Dashboard: 删除 canvas glow 死代码 (无效 hex→rgba)
- Dashboard: resize handler 加 rAF 节流
- Dashboard: IntersectionObserver 加 unobserve()
- Dashboard: header "8 Research Gaps" → "6 Research Entry Points" (与实际渲染一致)
- CLAUDE.md: Domain papers 路径从 `docs/papers/` 改为 `survey/papers/`

## v0.2.0 — 2026-04-11

### 新增
- **survey/papers/va-world-models-web.md** — WAM/Video WM/VA 最新论文 web 调研 (80+ 论文)，覆盖 DreamZero、NVIDIA Cosmos、DuoCore-FS 30Hz VLA、PocketDP3 等
- **viewer/static/survey-dashboard.html** — 交互式信息图 dashboard (Pareto frontier, pipeline comparison, maturity matrix, research gaps)
- **.claude/skills/project-skill/SKILL.md** — v0 bootstrap，7 章完整 context 文档 (overview, architecture, cognition, archive, experiments, lessons, reference)

### 修复
- 删除冗余 `experiments/` 目录，统一使用 `exp/`
- CLAUDE.md 目录结构描述补全实际目录
- Session startup 路径从空壳 `docs/papers/landscape.md` 改为 `survey/landscape.md`
- `.gitignore` 补充 Python/macOS/editor/ML artifacts 规则

## v0.1.0 — 2026-04-11

### 新增
- 项目初始化：LabMate 骨架 + git repo
- **survey/landscape.md** — VLM/VLA inference efficiency 全景 survey (80+ 论文)
- **survey/papers/recent-papers.md** — 2025-2026 最新论文 web 搜索 (40+ 论文)
- **survey/papers/va-world-models.md** — VA + World Action Model 深度分析 (60+ 论文)
- **notes/survey-plan.md** — 8 周调研行动计划
- CLAUDE.md 包含项目概述、survey 维度、LabMate workflow

### 其他
- selfOS wiki 同步更新：vlm-vla-real-time-systems concept 扩充、新建 world-action-model concept、方向启动 motivation source page
