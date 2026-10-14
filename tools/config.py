from pathlib import Path

# 项目根目录：config.py 位于 tools/ 下，上溯一级即项目根
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 允许「当纯文本读取」的文件后缀
TEXT_SUFFIXES = {".txt", ".md", ".py", ".json", ".csv", ".yaml", ".yml"}

# 文件列表/文件搜索里「可见」的后缀
LISTABLE_SUFFIXES = TEXT_SUFFIXES | {".pdf"}

# 不允许作为文档来源或操作目标的目录
BLOCKED_DIRECTORY_NAMES = {".git", ".venv", "venv", "__pycache__", "tests", ".pytest_cache"}

# 单次读取/写入的字节上限
MAX_FILE_SIZE_BYTES = 1_000_000

# 分块（chunker）相关参数
MAX_CHUNK_CHARS = 1200  # 单块字符上限；中文为主时建议降到约 500，避免超过 embedding 的 512 token 上限
OVERLAP_CHARS = 150     # 超长分块时相邻块的重叠字符数
MIN_CHUNK_CHARS = 40    # 低于此长度的块视为噪音丢弃