# Stage 1 归档：RAG 闭环（阶段 0 + 阶段 1）

> 这是一份交接文档：未来任何人（或新窗口）读完它，不需要重读历史对话，就能知道「做到哪了、怎么用、为什么这么设计、下一步做什么」。

## 一句话总结

从一个「5 个工具 + 简单 agent 循环」的学习项目，演进成了一个完整的 **Agentic RAG 学习助手**：PDF 转文本 → 切块 → 嵌入索引 → 语义问答 + 论文自动总结 + 全局术语表。

---

## 1. 完成度 + 现状

### 阶段 0（文档摄取）— 完成
- `pdf_to_text.py`：PDF → 结构化 Markdown（PyMuPDF + pymupdf4llm）；
- `config.py`：集中管理路径、文件类型、分块/索引参数；
- 支持类型：`.txt/.md/.py/.json/.csv/.yaml/.yml`（文本）；`.pdf` 由独立脚本 `pdf2md.py` 转成 `.md`（不经过 agent、也不出现在 list_documents 里）。

### 阶段 1（RAG 闭环）— 完成
- **M1.1**：`chunker.py`（结构感知分块）+ `embeddings.py`（bge 嵌入 + numpy 索引 + 余弦检索）+ `index.py`（构建入口）；
- **M1.2**：`search_knowledge.py`（语义检索工具）+ 接入 agent；
- **M1.3**：`summarizer.py`（map-reduce 总结 + 术语提取 + 去重 + 通用/专属分类）。

### 文件清单（每个一句话）

```text
根目录
├── agent.py              交互问答入口（Agentic RAG，6 个工具）
├── pdf2md.py             PDF → markdown 入口（离线脚本）
├── summarize.py          论文总结入口（离线流水线）
├── index.py              构建知识库索引入口
├── tool_schema.py        由 pydantic 自动生成 function schema（消除手写漂移）
├── glossary.md           全局术语表（只存通用术语，跨论文累积）
├── .env                  DEEPSEEK_API_KEY + DEEPSEEK_MODEL_NAME（勿提交）

tools/
├── config.py             集中配置（PROJECT_ROOT/CORPUS_DIRS/分块参数/模型名等）
├── file_reader.py        读取文本（read_file_content 原始 / read_file_tool 带行号）
├── file_searcher.py      列出/搜索文件（os.walk 剪枝，不遍历 .venv 等）
├── search_text.py        关键词检索（substring）
├── create_markdown.py / edit_markdown.py
├── pdf_to_text.py        PDF → markdown
├── chunker.py            结构感知分块（按标题，字符窗口兜底）
├── embeddings.py         嵌入 + 索引构建 + 余弦检索（numpy）
├── search_knowledge.py   语义检索工具
└── summarizer.py         论文总结流水线（map-reduce + 术语 + 去重）

article/  论文 PDF 与抽取出的 .md（语料）
notes/    学习笔记 + 设计草稿（语料）
knowledge/ 索引：chunks.json + vectors.npy（index.py 生成）
```

### 四个入口 + 工作流

```text
① 放论文      article/某论文.pdf
② PDF 转文本   python pdf2md.py article/某论文.pdf
③ 总结        python summarize.py "article/某论文.md"   → 生成 -summary.md，更新 glossary.md
④ 建索引      python index.py
⑤ 问答        python agent.py "这篇论文讲了什么"
```

> 环境：`.env` 配 `DEEPSEEK_API_KEY`、`DEEPSEEK_MODEL_NAME`；国内需 `setx HF_ENDPOINT "https://hf-mirror.com"`（新终端生效）。

---

## 2. 关键设计决策（为什么这么设计）

1. **分层边界**：内部模块（chunker/embeddings）↔ 工具（search_knowledge）↔ agent 循环。**agent 只通过工具、只传 query，接触不到内部函数和文件路径**——所以 agent 不会「检索根目录」。
2. **纯函数 vs IO 分离**：`read_file_content`（原始内容，供 search_text 内部复用）/ `read_file_tool`（带行号，供 agent 用）。切块逻辑 `chunk_text` 是纯函数（无副作用，便于测试）。
3. **可追溯**：每个 `Chunk` 带 `source_file + heading（面包屑）+ start_line/end_line`，检索命中后能回溯原文。
4. **路径安全**：`PROJECT_ROOT` + `BLOCKED_DIRECTORY_NAMES` + `os.walk` 剪枝——遍历时就不进入 `.venv/tests/__pycache__` 等。
5. **map-reduce 总结**：长文档（>20000 字）分节总结再汇总，短文档单次总结。
6. **术语从原文提取、保留英文原名**：这是去重能稳定的关键（翻译会导致同一个概念每次换说法）。
7. **归一化去重**：`term_key` = 小写 + 去括号注解 + 去所有分隔符（连字符/空格/斜杠）。
8. **策略 A**：通用术语进 `glossary.md`，论文专属名（KnowForge/Locator 这类）只留各自 summary。
9. **构建期 vs 查询期**：`index.py` 离线建索引（全量重建），`search` 在线查索引（只读），互不相干。

---

## 3. 关键常量（在 tools/config.py）

- `MAX_CHUNK_CHARS=1200`（bge 512 token 上限；中文为主建议降 ~500）、`OVERLAP_CHARS=150`、`MIN_CHUNK_CHARS=40`；
- `EMBEDDING_MODEL="BAAI/bge-small-zh-v1.5"`（512 维、中英双语）；
- `INDEX_DIR=knowledge/`、`CORPUS_DIRS=["article","notes"]`、`TOP_K=5`；
- `GLOSSARY_PATH=glossary.md`；
- summarizer 内：`SHORT_DOC_CHARS=20000`、`SECTION_MAX_CHARS=8000`。

---

## 4. 踩过的坑（防重踩）

1. **huggingface 被墙**：`setx HF_ENDPOINT "https://hf-mirror.com"`；且在代码里 `os.environ.setdefault("HF_ENDPOINT", ...)` 必须放在 **import sentence_transformers 之前**——因为 huggingface_hub 在 import 时就缓存了 endpoint。
2. **LLM 术语提取非确定**：同一篇论文两次提取术语名会变（翻译/加英文注解）。精确匹配有天花板 → 直接吃原文markdown，用「英文原名 + 归一化」解决。
3. **单行超长**：PDF 抽取会把整段糊成一行（>1200 字）。分块必须按「字符窗口」切（`_split_long`），不能只按行切。
4. **页眉/页脚污染**：每页重复的「CIKM '26, Rome, Italy」会进很多 chunk。**尚未处理**，记在 phase1 清单。

---

## 5. 下一步

已存清单：
- `notes/phase0-improvements.md`（文档摄取：docx/epub/编码/ipynb/表格/公式/图表/OCR…）
- `notes/phase1-improvements.md`（RAG：评估/混合检索/页眉去重/ChromaDB/增量/重排序…）
- `notes/two-pass-term-extraction.md`（超大论文的两段式术语提取，设计草稿未接入）

**第一梯队（建议下次做）**：① 检索评估 → ② 混合检索（关键词+语义 RRF 融合）→ ③ 页眉页脚去重。

**更大方向**：阶段 2（学习记忆 / 跨会话）、阶段 3（代码理解）。

---

## 6. 概念性收获（这个阶段内化的判断力）

- **Classic RAG vs Agentic RAG**：谁决定「查什么、查几轮」——代码写死 vs LLM 自主。
- **流水线 vs Agent Workflow vs LangGraph vs 多智能体**：
  - 流水线（summarizer.py）= 代码写死顺序，单 LLM 换 prompt，无状态、不回头；
  - Agent（agent.py）= LLM 自主决定调工具，有状态（messages 累积），但黑盒、不可靠、超轮次；
  - LangGraph = 图结构（节点+边）+ 状态 + 循环/分支 + 节点可做子代理，是「可控的编排」。
- **工具边界**：内部模块 ≠ agent 工具；agent 能碰的只有工具接口（query 参数），碰不到文件路径和内部函数。
- **固定批量任务用流水线，开放问答用 agent**——「总结论文」是前者，「交互问答」是后者，两者各取所长。
