# StudyFlow Lite

一个基于本地文档的 **Agentic RAG 学习助手**：读论文、建索引、混合检索问答、引导式精读，并自动维护术语表。

最初是为 Kaggle「5-Day AI Agents Intensive Course」的学习笔记而建，现已逐步扩展成一个完整的 RAG 项目。

## 功能

- **论文精讲陪读**：按 `##` 分节逐节讲解，讲完一节停下等你发话；你随时针对当前内容提问；退出后保留进度，下次可续读（`reader.py`）；
- **论文总结（存笔记）**：精读中随时说「存笔记」，生成结构化笔记（一句话总结 / 研究问题 / 方法 / 贡献 / 关键技术），并把术语去重后写入全局术语表；
- **概念复习**：精读中说「复习 <概念>」，从本地知识库混合检索后重新讲解；结果不写回阅读上下文，避免打断连续性；
- **混合检索**：dense（bge 嵌入）+ sparse（TF-IDF 词法）+ RRF 名次融合，兼顾「语义相近」与「精确术语 / 跨表述」；
- **文档问答（Agentic RAG）**：agent 自主决定查什么、查几轮，基于混合检索带出处回答；
- **PDF 转文本**：把 PDF 论文抽成结构化 Markdown（支持双栏、表格与标题）；
- **知识库索引**：本地 numpy 索引（chunks.json + vectors.npy + sparse.json），构建一次、反复查询；
- **文件工具**：读取、列表、关键词搜索、创建 / 编辑 Markdown。

## 入口对照

| 入口 | 角色 | 什么时候用 |
|---|---|---|
| `pdf2md.py` | 摄取 | 把 PDF 论文转成 Markdown |
| `index.py` | 索引 | 语料变了，供检索 / 复习 / agent 问答使用 |
| `reader.py` | **精读（workflow）** | 坐下来逐节读一篇、随问随答、存笔记 |
| `Agentic RAG.py` | **研究（agent）** | 跨论文、多轮检索、自由问答 |

> `reader.py` 是「工作流」：流程由固定命令控制，模型只负责讲解与回答。`Agentic RAG.py` 是「智能体」：由模型自主决定调用哪些工具。二者互补，不是重复。

## 目录结构

```text
studyflow-lite/
├── reader.py             # 引导式精读入口（学 / 下一节 / 存笔记 / 复习 / exit）
├── Agentic RAG.py        # 交互问答入口（Agentic RAG）
├── pdf2md.py             # PDF → Markdown 入口（离线脚本）
├── index.py              # 构建知识库索引入口
├── tool_schema.py        # 由 pydantic 自动生成 function schema
├── glossary.md           # 全局术语表（跨论文累积）
├── project-ideas.md      # 项目想法
├── .env                  # API key 与模型名（勿提交 git）
├── article/              # 论文 PDF 与抽取出的 .md（语料）
├── notes/                # 学习笔记（语料）
├── knowledge/            # 索引 + 阅读进度（index.py / reader.py 生成）
└── tools/                # 工具模块
    ├── config.py         # 集中配置（语料目录、索引路径、模型名等）
    ├── file_reader.py    # 读取文本文件
    ├── file_searcher.py  # 列出/搜索文件
    ├── search_text.py    # 关键词检索
    ├── create_markdown.py / edit_markdown.py
    ├── pdf_to_text.py    # PDF → markdown
    ├── chunker.py        # 结构感知分块
    ├── dense_search.py / sparse_search.py / hybrid_search.py  # 混合检索
    ├── section_parser.py # 按 ## 分节（精读导航）
    ├── reading_session.py# 精读状态 + 命令 + prompt 装配
    └── summarizer.py     # 总结生成 + 落盘（由「存笔记」调用）
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
- 之后用 `reader.py` 精读、或 `index.py` 建索引。

### 2. `reader.py` —— 引导式精读

```powershell
.venv\Scripts\python.exe reader.py "article/Role-Specialized Tool Agents.md"
```

- 作用：按 `##` 分节逐节讲解；你随时提问；「下一节」推进；「存笔记」生成 `-summary.md` 并更新 `glossary.md`；「复习 <概念>」走知识库混合检索（结果不写回阅读上下文）；`exit` 退出并保存进度，下次可续读；
- 命令：`学 <论文>` / `下一节` / `存笔记` / `复习 <概念>` / `exit`；
- 前提：先把 PDF 转成 `.md`（用上面的 pdf2md.py）；「复习」需要先跑过 `index.py`；
- 注意：会把整篇论文放进上下文，且「存笔记」会多次调用模型，**消耗真实 API 额度**。

### 3. `index.py` —— 构建知识库索引

```powershell
.venv\Scripts\python.exe index.py
```

- 作用：把 `article/` 和 `notes/` 下的文档**切块 → 向量化 → 存入 `knowledge/`**；
- 何时跑：**语料变了就重跑**（新增论文/笔记后），属全量重建、会覆盖旧索引；
- 想加新语料目录：在 `tools/config.py` 的 `CORPUS_DIRS` 里加目录名即可。

### 4. `Agentic RAG.py` —— 交互问答（Agentic RAG）

> 文件名含空格，命令里需要加引号。

```powershell
# 交互式
.venv\Scripts\python.exe "Agentic RAG.py"

# 直接带问题启动
.venv\Scripts\python.exe "Agentic RAG.py" "什么是 RAG"
```

- 作用：多轮问答，agent 自主决定调用哪些工具（混合检索 / 关键词 / 读文件 / 写 Markdown 等）；输入 `exit` 退出；
- 前提：先跑过 `index.py`，否则 `hybrid_search` 会提示「索引尚未构建」。

## 典型工作流

```text
① 放论文        article/某论文.pdf
② PDF 转文本     python pdf2md.py article/某论文.pdf
                （生成同名 .md）
③ 精读 + 存笔记  python reader.py "article/某论文.md"
                → 逐节讲解 → 存笔记生成 -summary.md、更新 glossary.md
④ 建索引        python index.py        （供检索 / 复习 / agent 用）
⑤ 研究 / 问答    python "Agentic RAG.py" "这篇论文讲了什么"
```

> PDF 转换是独立脚本；精读走 `reader.py`；跨论文研究走 `Agentic RAG.py`。

## 许可

本项目采用 **PolyForm Noncommercial License 1.0.0**（源码公开、禁止商业使用）：

- ✅ 允许：个人学习、研究、教学等**非商业目的**下使用、修改、分发；
- ❌ 禁止：任何**商业用途**；
- 分发时需附带本许可条款。

> 说明：这不是 OSI 意义上的「开源许可」（OSI 要求不限制使用领域），而是「source-available（源码公开）」许可。完整条款见 [LICENSE](LICENSE)。

## 个人感想

开始这个项目是因为大四开始找工作找实习了，但是有关我这个人工智能的专业的岗位有一大堆我在校期间根本没接触的东西，或者说是只是说过但却从来没有展开来教的东西，比如：skill啊，agent啊，rag啊这些，学校教的更多的是原理，什么是ai，如何去造一个ai，但没教怎么去用ai，怎么去使用和构建一个智能体（也就是agent）。

所以我开始感到疑惑和焦虑，感觉自己好像这4年啥也没学会，和社会需求好像脱节了，虽然我知道学习了大学本科的这些课程并不是完全无用，它让我有了这些基础，能知道原理，能看懂一些代码，也能在之后进行科研的时候，真的有机会自己创造一个属于自己的ai的时候会派上用场，但是就目前来说也许太过遥远，但是我依旧会保有这份渴望。

但幸运的是，大学真正教会我并让我觉得能受益一生的便是自主学习的能力，大学大学，大不了自学，也许我的编码水平，算法能力可能不如其他计算机专业的同学强，但是我依旧在努力学习，只有保有这颗持续探索，学习，并对这一切好奇的心，我想我也能在这样的时代下让自己继续成长，变得足够强大，有朝一日能坦坦荡荡得面对最初的理想。

于是，我开始了......
