from pathlib import Path

from pydantic import BaseModel, Field, model_validator


# 工具默认只能读取项目根目录中的文件，避免意外访问项目外的路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 只允许读取常见文本格式；PDF、Word 等格式需要专门的解析库。
SUPPORTED_SUFFIXES = {".txt", ".md", ".py", ".json", ".csv", ".yaml", ".yml"}

# 限制单次读取大小，避免误读大型文件占用过多内存或上下文。
MAX_FILE_SIZE_BYTES = 1_000_000

# 这些目录通常包含依赖、版本控制数据或缓存，不应作为文档来源。
BLOCKED_DIRECTORY_NAMES = {".git", ".venv", "venv", "__pycache__"}


class ReadFileInput(BaseModel):
	"""读取文件工具的结构化输入。"""

	file_path: str = Field(
		description="项目根目录下的相对文本文件路径，不允许绝对路径，例如 notes/day-1.md。"
	)
	start_line: int | None = Field(
		default=None,
		ge=1,
		description="可选的起始行号，从 1 开始；省略时从文件开头读取。",
	)
	end_line: int | None = Field(
		default=None,
		ge=1,
		description="可选的结束行号，包含该行；省略时读取到文件末尾。",
	)

	@model_validator(mode="after")
	def validate_line_range(self) -> "ReadFileInput":
		"""确保起止行号有效，并且文件路径不是空字符串。"""
		if not self.file_path.strip():
			raise ValueError("file_path 不能为空")
		if (
			self.start_line is not None
			and self.end_line is not None
			and self.end_line < self.start_line
		):
			raise ValueError("end_line 不能小于 start_line")
		return self


def read_file_tool(request: ReadFileInput) -> str:
	"""读取项目内的 UTF-8 文本文件，可选按 1 起始的行号范围截取。"""
	requested_path = Path(request.file_path)

	# 不接受绝对路径，工具调用方只能指定项目内的相对路径。
	if requested_path.is_absolute():
		raise ValueError("请提供项目根目录下的相对路径")

	# resolve 会处理 .. 和符号链接；relative_to 用来确认最终路径仍在项目内。
	resolved_path = (PROJECT_ROOT / requested_path).resolve()
	try:
		resolved_path.relative_to(PROJECT_ROOT)
	except ValueError as error:
		raise ValueError("不允许读取项目目录以外的文件") from error

	# 阻止读取依赖、缓存和版本控制目录中的文件。
	if BLOCKED_DIRECTORY_NAMES.intersection(resolved_path.parts):
		raise ValueError("不允许读取依赖、缓存或版本控制目录中的文件")

	# 用扩展名限制文件类型，避免把二进制文件当作文本读取。
	if resolved_path.suffix.lower() not in SUPPORTED_SUFFIXES:
		raise ValueError(
			f"不支持的文件类型：{resolved_path.suffix or '无扩展名'}；"
			f"支持类型：{', '.join(sorted(SUPPORTED_SUFFIXES))}"
		)

	if not resolved_path.is_file():
		raise FileNotFoundError(f"文件不存在：{request.file_path}")

	# 先检查大小，再读取内容，避免意外加载过大的文件。
	if resolved_path.stat().st_size > MAX_FILE_SIZE_BYTES:
		raise ValueError(
			f"文件超过 {MAX_FILE_SIZE_BYTES} 字节的读取上限：{request.file_path}"
		)

	# 显式使用 UTF-8，确保不同 Windows 区域设置下的读取行为一致。
	content = resolved_path.read_text(encoding="utf-8")
	lines = content.splitlines(keepends=True)

	# Python 切片的结束位置不包含该位置；这里把用户的闭区间行号转成切片。
	start_index = (request.start_line or 1) - 1
	end_index = request.end_line if request.end_line is not None else len(lines)
	return "".join(lines[start_index:end_index])
