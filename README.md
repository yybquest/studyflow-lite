# StudyFlow Lite

一个基于本地文档的 **Agentic RAG 学习助手**：读论文、建索引、语义问答、自动总结并维护术语表。

最初是为 Kaggle「5-Day AI Agents Intensive Course」的学习笔记而建，现已逐步扩展成一个完整的 RAG 项目。

## 功能

- **文档问答（Agentic RAG）**：agent 自主决定查什么、查几轮，基于本地知识库语义检索后带出处回答；
- **论文总结**：一条命令把论文整理成结构化笔记（一句话总结 / 研究问题 / 方法 / 贡献 / 关键技术）+ 自动维护全局术语表；
- **PDF 转文本**：把 PDF 论文抽成结构化的 Markdown；
- **语义检索**：bge 嵌入 + 余弦相似度，支持跨表述的「意思相近」检索；
- **知识库索引**：本地 numpy 索引（chunks.json + vectors.npy），构建一次、反复查询；
- **文件工具**：读取、列表、关键词搜索、创建/编辑 Markdown。

## 目录结构

```text
studyflow-lite/
├── agent.py              # 交互问答入口（Agentic RAG）
├── pdf2md.py             # PDF → Markdown 入口（离线脚本）
├── summarize.py          # 论文总结入口（离线流水线）
├── index.py              # 构建知识库索引入口
├── tool_schema.py        # 由 pydantic 自动生成 function schema
├── glossary.md           # 全局术语表（跨论文累积）
├── project-ideas.md      # 项目想法
├── .env                  # API key 与模型名（勿提交 git）
├── article/              # 论文 PDF 与抽取出的 .md（语料）
├── notes/                # 学习笔记（语料）
├── knowledge/            # 索引：chunks.json + vectors.npy + sparse.json（index.py 生成）
└── tools/                # 工具模块
    ├── config.py         # 集中配置（语料目录、索引路径、模型名等）
    ├── file_reader.py    # 读取文本文件
    ├── file_searcher.py  # 列出/搜索文件
    ├── search_text.py    # 关键词检索
    ├── create_markdown.py / edit_markdown.py
    ├── pdf_to_text.py    # PDF → markdown
    ├── chunker.py        # 结构感知分块
    ├── dense_search.py     # 嵌入 + 索引 + 语义检索
    └── summarizer.py     # 论文总结流水线
```

## 环境准备

1. **依赖**（装进虚拟环境）：

   ```powershell
   python -m pip install openai python-dotenv pydantic numpy sentence-transformers pymupdf pymupdf4llm
   ```

2. **配置 `.env`**（在项目根目录）：

   ```text
   DEEPSEEK_API_KEY="你的 key"
   DEEPSEEK_MODEL_NAME="deepseek-flash"
   ```

3. **（国内）设置 HuggingFace 镜像**，否则首次下载 bge 嵌入模型会失败：

   ```powershell
   setx HF_ENDPOINT "https://hf-mirror.com"
   ```

   设置后**新开一个终端**再继续。模型只下载一次，之后从本地缓存加载。

## 使用说明（四个入口）

### 1. `pdf2md.py` —— PDF 转 Markdown（离线脚本）

```powershell
.venv\Scripts\python.exe pdf2md.py "article/某论文.pdf"
```

- 作用：把 PDF 抽成结构化 Markdown，生成同名 `.md`；
- 之后用 `summarize.py` 总结、或 `index.py` 建索引。

### 2. `summarize.py` —— 总结论文（离线流水线）

```powershell
.venv\Scripts\python.exe summarize.py "article/Role-Specialized Tool Agents.md"
```

- 作用：map-reduce 生成结构化笔记，写到 `同目录/<原名>-summary.md`；术语提取后，**通用术语**进 `glossary.md`、**本文专属名词**只留在 summary；
- 前提：先把 PDF 转成 .md（用上面的 pdf2md.py）；
- 注意：会调用多次 LLM，**消耗真实 API 额度**，建议先拿短文档试跑。

### 3. `index.py` —— 构建知识库索引

```powershell
.venv\Scripts\python.exe index.py
```

- 作用：把 `article/` 和 `notes/` 下的文档**切块 → 向量化 → 存入 `knowledge/`**；
- 何时跑：**语料变了就重跑**（新增论文/笔记后），属全量重建、会覆盖旧索引；
- 想加新语料目录：在 `tools/config.py` 的 `CORPUS_DIRS` 里加目录名即可。

### 4. `agent.py` —— 交互问答（Agentic RAG）

```powershell
# 交互式
.venv\Scripts\python.exe agent.py

# 直接带问题启动
.venv\Scripts\python.exe agent.py "什么是 RAG"
```

- 作用：多轮问答，agent 自主决定调用哪些工具（语义检索 / 关键词 / 读文件 / 写 Markdown 等）；输入 `exit` 退出；
- 前提：先跑过 `index.py`，否则 `hybrid_search` 会提示「索引尚未构建」。

## 典型工作流

```text
① 放论文        article/某论文.pdf
② PDF 转文本     python pdf2md.py article/某论文.pdf
               （生成同名 .md）
③ （可选）总结   python summarize.py "article/某论文.md"
               → 生成 -summary.md，更新 glossary.md
④ 建索引        python index.py
⑤ 问答          python agent.py "这篇论文讲了什么"
```

> PDF 转换是独立脚本（不走 agent）；agent 只负责最后一步的交互问答。

## 个人感想

开始这个项目是因为大四开始找工作找实习了，但是有关我这个人工智能的专业的岗位有一大堆我在校期间根本没接触的东西，或者说是只是说过但却从来没有展开来教的东西，比如：skill啊，agent啊，rag啊这些，学校教的更多的是原理，什么是ai，如何去造一个ai，但没教怎么去用ai，怎么去使用和构建一个智能体（也就是agent）。

所以我开始感到疑惑和焦虑，感觉自己好像这4年啥也没学会，和社会需求好像脱节了，虽然我知道学习了大学本科的这些课程并不是完全无用，它让我有了这些基础，能知道原理，能看懂一些代码，也能在之后进行科研的时候，真的有机会自己创造一个属于自己的ai的时候会派上用场，但是就目前来说也许太过遥远，但是我依旧会保有这份渴望。

但幸运的是，大学真正教会我并让我觉得能受益一生的便是自主学习的能力，大学大学，大不了自学，也许我的编码水平，算法能力可能不如其他计算机专业的同学强，但是我依旧在努力学习，只有保有这颗持续探索，学习，并对这一切好奇的心，我想我也能在这样的时代下让自己继续成长，变得足够强大，有朝一日能坦坦荡荡得面对最初的理想。

于是，我开始了......
