# Shell Script Safety Patterns

> SSH 远程命令注入防护、pipefail、Flask debug 模式等研究脚本安全模式

## Problem
研究项目的 convenience scripts (launch, remote, viewer) 常被 Codex/code review 标记安全问题。虽然是内部工具，但养成好习惯避免生产项目犯同样错误。

## Cause
- SSH 命令拼接中变量未验证 → command injection
- `set -e` 不覆盖 pipe 中的失败 → 静默失败
- Flask `debug=True` 硬编码 → Werkzeug debugger RCE

## Solution

### 1. SSH 远程命令 — Input Validation (allowlist)

引号方案 (`'${VAR}'`) 在双引号 SSH 字符串中不够安全（`'` 可逃逸）。正确做法是 input validation：

```bash
if [[ "$GPU_ID" =~ [^0-9] ]] || [[ "$CONFIG_NAME" =~ [^a-zA-Z0-9/_.-] ]]; then
    echo "ERROR: Invalid characters in GPU_ID or CONFIG_NAME" >&2
    exit 1
fi
ssh host "command ${GPU_ID} ${CONFIG_NAME}"
```

### 2. Pipefail — 管道中的错误传播

```bash
set -eo pipefail
# 现在 `python cmd | tee log` 中 python crash 会正确传播
```

不加 `pipefail` 时，只有管道最后一个命令的 exit code 被 `set -e` 检查。

### 3. Flask debug 模式 — 环境变量控制

```python
# 默认关闭，通过环境变量启用
import os
app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1", port=5001)
```

开发启动脚本中设置：
```bash
FLASK_DEBUG=1 python viewer/app.py
```

## Notes
- Date: 2026-04-20
- Context: Codex adversarial review 两轮指出的安全问题
- 适用范围: 所有 scripts/ 下的 shell 脚本和 Flask viewer
