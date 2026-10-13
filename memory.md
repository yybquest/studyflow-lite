# Memory（经验与决策记录）

> 这个文件记录我在做这个 agent 项目过程中踩过的坑、做过的决策、以及沉淀下来的经验教训。
> 目的：以后回顾时不用重踩坑；也让未来的 agent 能读到这些经验。
> 与 `skill.md` 的区别：memory 记「发生过什么、为什么」，skill 记「下次照着怎么做」。

---

## 条目 1 · PDF 文本抽取：从双栏乱码到 pymupdf4llm

- 记录时间：2026-10-13
- ⚠️ 更正：本条目里「pymupdf4llm 是布局启发式」的理解后来被推翻——它其实默认走 GNN（神经网络）布局引擎，见下方「条目 2」。

### 背景

要把 PDF 论文抽成文本喂给学习助手。最初用 PyMuPDF 的 `page.get_text(sort=True)`。

### 遇到的问题

双栏论文抽取结果混乱：

- 左、右栏同一高度的文字被拼成一行，中间夹一大串空格（那串空格是 PyMuPDF 用来还原横向距离的占位）；
- 页眉碎片（如 `[cs.AI]`）被拆成独立文本块后混进正文；
- 表格结构丢失、图片内容丢失（只剩图注）。

### 试过的方案与结论

| 方案 | 结果 | 原因 / 说明 |
| --- | --- | --- |
| `get_text(sort=True)` | 双栏乱码 | 只是按坐标粗略排序，不识别分栏，不等于阅读顺序还原 |
| `find_tables()` lines 策略 | 把条形统计图误判成表格 | 靠「线」检测，图的坐标轴/网格线被当成表格线 |
| `find_tables()` text 策略 | 把双栏正文误判成表格 | 靠「文字对齐」检测，分栏正文天然对齐成列 |
| `pymupdf4llm.to_markdown()` | ✅ 清晰可用 | GNN 布局模型（默认启用），正确处理分栏 / 标题 / 表格；详见条目 2 |

### 最终方案

用官方 `pymupdf4llm.to_markdown(doc)` 替代 `get_text(sort=True)`，输出结构化 Markdown。

### 沉淀下来的教训

1. **喂给 LLM 的内容，结构化 Markdown 永远优于扁平纯文本。**
2. `get_text(sort=True)` 是「几何排序」，不是「阅读顺序还原」；双栏论文必须用布局感知的转换。
3. 手调的 `find_tables()`（纯几何法）对「图 / 表 / 分栏正文」高度混淆，不可靠；但 pymupdf4llm 内置的 GNN 表格模型（table_grid）能做得更好，不必等「多模态」——见条目 2。
4. PDF 用 ingest 模式（`pdf_to_text` 抽成 .md 入库 → 复用 read/search 工具），优于现场直读。
5. 选型要实测多种方案、看清各自的失败模式、基于证据做决定，而不是「能跑就行」。

### 一句话原则

> 摄取层（PDF → 文本）应该输出「结构化 Markdown」，而不是「扁平纯文本」。

---

## 条目 2 · 纠正：pymupdf4llm 其实是「神经网络（GNN）」驱动，不是纯启发式

- 记录时间：2026-10-13

### 背景

条目 1 里把 `pymupdf4llm` 当成了「几何 + 排版启发式」工具。后来查环境里安装的源码和 METADATA，发现这个理解对 1.28.2 版本是**错的**。

### 查证到的真相

- `pymupdf4llm` 1.28.2 的 METADATA 里把 `pymupdf_layout` 列为**强制依赖**（`Requires-Dist: pymupdf_layout==1.28.2`），所以 `pip install pymupdf4llm` 会自动装进 `pymupdf-layout`。
- `pymupdf-layout` 往 `pymupdf` 包里注入了 `pymupdf/layout/`，里面是一组 **ONNX 小模型**：
  - `layout_rf2.4.1.onnx`：版面分类（GNN，代码里叫 `BoxRFDGNN`）；
  - `table_grid_model_v1 ~ v4.onnx`：表格网格预测（`TableGridExtractor`）；
  - `feature_imf1/imf2.onnx`：图像特征。
- 加起来几十 MB，本地、CPU-only（ONNX Runtime）、无需 GPU、数据不出本机。

### 默认怎么被调起来

- `pymupdf4llm/__init__.py` 默认尝试 `import pymupdf.layout`，成功就 `use_layout(True)`；
- `to_markdown()` 检测到 `_use_layout=True` → 走 `_layout_to_markdown` → `document_layout.parse_document` → 每页调 `page.get_layout()`（GNN 推理）；
- GNN 输出带类别标签的 box（`title` / `section-header` / `text` / `table` / `picture` / `formula` / `footnote` / `list-item` …），再由**确定性代码**按类别序列化成 Markdown；
- 只有 `import pymupdf.layout` 失败（没装 pymupdf-layout）时才 fallback 到旧的启发式 `pymupdf_rag` 路径。

### 关键理解

1. **模型负责「看懂版面」，代码负责「拼 Markdown」**——不是端到端。
2. 表格识别之所以比手调的 `find_tables()` 强，是因为用了学出来的 `table_grid` 模型（GridPrediction），不是几何启发式。
3. `pymupdf-layout` 是「AI trained on PDF internals, not images」——GNN 直接在 PDF 的文字/向量图元上建图，不渲染成图片，所以 CPU-only、快约 10 倍、无需 GPU。
4. 架构对比：视觉方案（渲染成图 → 视觉模型）vs 结构方案（PDF 图元 → 建图 → GNN）。pymupdf-layout 属于后者。

### 沉淀下来的教训

1. **库会演进，凭记忆的认知会过时**；「读安装包里的源码和 METADATA」才是 ground truth。
2. 这正是「大模型（agent）调用专用小模型当工具」的活例子——agent 不会自己分析版面，而是通过 `pdf_to_text` 工具调用本地的布局小模型。
