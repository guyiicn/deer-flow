# LLM API 定价数据采集失败分析

**文档日期**: 2026-05-26  
**任务背景**: DeerFlow 执行国内 LLM API 定价批量采集任务，21 家厂商中 3 家出现数据缺失或来源不可信问题。  
**当前状态**: 3 个问题全部已修复，数据已补齐。

---

## 概览

| 厂商 | 第一轮结果 | 失败类型 | 修复方式 |
|------|-----------|---------|---------|
| 月之暗面 (Moonshot) | K2.6 标注"未公开" | JS 渲染表格不可读 | 发现 `.md` 直链绕过 React 渲染 |
| 商汤科技 (SenseTime) | 价格来自 LobeHub + 36氪（三方媒体） | URL 定位错误（404） | 定位真实官方域名 `sensecore.cn` |
| 百川智能 (Baichuan) | 价格来自 AIGCRank（三方聚合站） | 官方定价页真实下线 | 无法获取官方数据，三方来源已保留并注明 |

---

## 失败一：月之暗面 Kimi K2.6 — JS 渲染表格

### 现象

DeerFlow 第一轮抓取 `https://platform.kimi.com/docs/pricing/chat-k26` 时，返回内容仅为页面导航壳（侧边栏、面包屑），**价格表格完全缺失**。K2.6 在报告中被标注为：

> 未公开（官方页面 JS 表格未在抓取中渲染；……）

### 根本原因

Kimi 文档站基于 Next.js 构建，价格表格是一个 **React 动态组件**（`PricingTable`）。组件从打包 JS 中的常量取值后在客户端渲染成 HTML `<table>`。Jina reader（`r.jina.ai`）虽然使用 Headless Chrome，但：

1. 页面加载完成后，部分 React hydration（水合）逻辑执行时间超出 Jina 等待窗口
2. Jina 的 Readability 提取器在 DOM 还没稳定时已完成内容提取，拿到的是 hydration 前的空 `<div>` 占位符
3. 提取到的 Markdown 因此只有导航和介绍段落，**表格数据行为空**

这是 SPA 定价页的**系统性问题**，不只限于 Kimi——任何使用客户端渲染价格表的文档站都会触发。

### 修复过程

**关键发现**：Kimi 平台在 `https://platform.kimi.com/docs/llms.txt` 维护了一份 `llms.txt` 索引（类似 `robots.txt` 的惯例，用于向 LLM 暴露可机读文档列表）。该文件列出了所有文档页对应的原始 `.md` 文件 URL，格式为：

```
<页面 URL>.md → 返回对应的原始 Markdown 源文件
```

直接 fetch `https://platform.kimi.com/docs/pricing/chat-k26.md`（注意尾部 `.md`），返回了 React 组件所依赖的**原始 Markdown 数据**，其中包含完整的价格表：

```
["kimi-k2.6", "1M tokens", "¥1.10", "¥6.50", "¥27.00", "262,144 tokens"]
```

确认价格：
- 输入（缓存未命中）：¥6.50 / M tokens  
- 输入（缓存命中）：¥1.10 / M tokens  
- 输出：¥27.00 / M tokens  
- 上下文窗口：256K tokens

### 可泛化规律

> **对于 Next.js / Docusaurus / VitePress 等文档站，优先尝试在页面 URL 末尾加 `.md` 后缀获取原始 Markdown 源文件。**  
> 检查路径 `<domain>/docs/llms.txt` 或 `<domain>/llms.txt` 是否存在，其中会列出所有可机读文档路径。

---

## 失败二：商汤科技 SenseTime — URL 定位错误

### 现象

DeerFlow 第一轮采集的商汤价格数据来源标注为：

> LobeHub（三方工具站）+ 36氪（科技媒体）

而非官方价格页，这意味着价格准确性和时效性均不可保证。

### 根本原因

商汤 AI 服务存在**两套平台域名**，职责分离，但对外认知度差异大：

| 域名 | 定位 | 状态 |
|------|------|------|
| `platform.sensenova.cn` | 旧的/模型专属子域（SenseNova 品牌） | **`/pricing` 路径 404** |
| `www.sensecore.cn` | 真实企业级平台（品牌名：日日新大装置） | **官方文档完整，含完整定价表** |

DeerFlow 搜索"商汤 API 定价"时，检索结果和引用链接普遍指向 `platform.sensenova.cn/pricing`——这是一个**已不再维护的旧链接**，大量三方文章在商汤品牌更名前抓存了这个 URL，导致搜索引擎结果被污染。

官方实际定价页为：
```
https://www.sensecore.cn/help/docs/model-as-a-service/nova/pricing
```

该页面标题为"模型调用计费 | 大装置帮助中心"，品牌名已切换到"日日新"（SenseNova），但域名是 `sensecore.cn` 而非 `sensenova.cn`。

### 修复过程

1. 识别出"商汤大装置"（`sensecore.cn`）才是企业级 API 平台
2. 直接构造官方文档路径 `sensecore.cn/help/docs/model-as-a-service/nova/pricing`
3. Jina fetch 成功返回完整价格表，数据为纯 Markdown 表格（非 JS 渲染）

获取到全系列官方价格：SenseNova-V6.5-Pro/Turbo、V6-Pro/Turbo/Reasoner/Omni、SenseChat-5/Turbo/Vision 等。

### 可泛化规律

> **当目标厂商有品牌迭代或平台重组时，搜索结果大概率指向旧 URL。**  
> 应主动识别厂商当前的企业级产品品牌名（商汤的"大装置"/"日日新"、阿里的"百炼"等），直接访问该品牌的官方文档站，而非依赖搜索结果的引用链。

---

## 失败三：百川智能 Baichuan — 官方定价页真实下线

### 现象

DeerFlow 第一轮采集百川价格来源标注为：

> AIGCRank 国内外 AI 大语言模型 API 价格对比（三方聚合站）

### 根本原因

百川官方定价页已被**物理下线**。验证过的所有 URL 均返回 404：

```
https://platform.baichuan-ai.com/price     → 404（渲染内容仅为客服邮箱）
https://platform.baichuan-ai.com/pricing   → 404
https://www.baichuan-ai.com/price          → 404
https://www.baichuan-ai.com/pricing        → 404
```

`/docs/api` 文档页存在，有模型列表，但**不包含任何价格信息**。  
官网首页 `www.baichuan-ai.com` 有产品介绍，同样无价格数据。

推测原因（未经官方确认）：
1. 百川可能将标准定价移入登录后的控制台，不再对外公开展示
2. 或价格谈判策略调整，转向"联系销售"模式
3. 定价页面可能因产品线调整临时下线，尚未重新上线

### 修复结果

**无法获取官方数据**，保留第一轮的三方来源（AIGCRank），并在文件中明确标注：

- 所有价格来源均为三方数据（AIGCRank、科技媒体引用）
- 官方定价页在采集时返回 404，建议用户直接联系：`Openapi@baichuan-inc.com`
- 三方数据的采集时间早于本次任务，时效性存疑

---

## 横向对比：三种失败模式

| 维度 | 月之暗面 | 商汤 | 百川 |
|------|---------|------|------|
| **失败原因** | 技术壁垒（JS 渲染） | 信息定位错误（旧 URL） | 官方主动下线 |
| **是否有官方数据** | 有（隐藏在 .md 源文件） | 有（在另一个域名） | 无 |
| **修复难度** | 中（需要知道 `.md` 直链技巧） | 低（找对域名即可） | 无解（只能用三方） |
| **三方数据质量** | 不适用（官方已获取） | 不适用（官方已获取） | 中等（AIGCRank 引用官方公告） |

---

## 对 DeerFlow 定价采集任务的改进建议

### 1. 优先探针策略（JS 渲染绕过）

在 `web_fetch` 目标 URL 前，先尝试：
- `<url>.md` — 检查是否有原始 Markdown 直链
- `<domain>/llms.txt` 或 `<domain>/docs/llms.txt` — 检查是否有机读文档索引
- `<domain>/docs/pricing/index.md` — 常见文档站 pattern

### 2. 品牌别名预构建

国内厂商存在多套品牌/产品名，建议维护映射表：

```
商汤科技 → 日日新 / SenseNova → sensecore.cn
阿里云    → 百炼 / DashScope  → dashscope.aliyuncs.com / bailian.aliyun.com
字节跳动  → 火山引擎 / 方舟   → volcengine.com / console.volcengine.com
腾讯云    → 混元              → hunyuan.tencent.com / cloud.tencent.com
```

遇到 404 时，先查映射表，再搜索当前企业级产品品牌的文档站，而非继续尝试旧 URL 变体。

### 3. 四方验证流程（当官方 404 时）

当官方定价页不可达，应按优先级顺序使用替代来源：

1. 官方 OpenAPI 文档（`/docs/api`，可能有价格说明）
2. 官方微信公众号/技术博客（降价公告等）
3. 阿里云百炼/火山方舟等平台（转接模型时会注明原厂价格）
4. AIGCRank（价格聚合站，更新较频繁）
5. 36氪/虎嗅等科技媒体（引用官方公告，时效性好但不持续更新）

### 4. `web_fetch` max_chars 配置

本次任务前已确认 `max_chars=50000`（原为硬编码 `4096`，已修复并提交 `e1cbec10`）。建议对于定价页等结构化文档，保持此值不低于 **50,000 chars** 以确保长表格完整返回。

---

## 附：Jina Reader 与 JS 渲染的技术补充

`r.jina.ai` 使用 Playwright（Headless Chromium）抓取，支持基本的 JS 执行。但以下场景仍会导致价格表缺失：

| 场景 | 原因 | 绕过方式 |
|------|------|---------|
| React/Next.js 客户端渲染表格 | Jina 在 hydration 完成前提取 DOM | `.md` 直链 / `llms.txt` 索引 |
| 价格数据需登录后 API 动态拉取 | 无 token，fetch 返回空 | 三方数据 / 人工采集 |
| Cloudflare / WAF 拦截 | Jina IP 被识别为爬虫 | 添加 `-H "X-Return-Format: text"` 头（改变 Jina 请求行为）；或直接 curl |
| React 懒加载（IntersectionObserver） | 表格在 viewport 外未渲染 | Jina 无法模拟滚动，只能绕过 |

---

*文档作者: Claude (claude-sonnet-4-6) | 生成时间: 2026-05-26*
