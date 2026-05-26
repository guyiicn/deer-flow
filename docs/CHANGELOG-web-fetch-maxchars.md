# Fix: web_fetch 截断限制 (4096 → 可配置 max_chars)

**日期**: 2026-05-26  
**分支**: phase4-g2-enhancement

## 问题

`web_fetch` 工具在返回页面内容时硬编码了 `[:4096]` 截断，导致：
- 长页面（如 aliyun 定价页 143,665 字符）只返回前 4096 字符
- Agent 无法读到表格中段、后段的定价数据
- 研究任务中大量"未公开"实为截断导致的数据缺失

## 修改

### `backend/packages/harness/deerflow/community/jina_ai/tools.py`

- 新增 `max_chars` 参数读取（从 `config.yaml` tool 配置的 `model_extra` 中读取）
- 默认值 50,000 字符（原来 4,096）
- 保持 error passthrough 逻辑不变

### `config.yaml`

- 在 `web_fetch` tool 条目中新增 `max_chars: 50000`
- 可按需调整（如改为 100000）

## 验证

```
aliyun model-pricing 页面:
  旧行为: 4,096 chars (仅页面开头，无完整定价表)
  新行为: 143,665 chars (全量内容，含所有模型定价行)
```
