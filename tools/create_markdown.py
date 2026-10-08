from pathlib import Path

from pydantic import BaseModel, Field, model_validator

# 工具默认只能读取项目根目录中的文件，避免意外访问项目外的路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 限制单次最大写入字节上限
MAX_FILE_SIZE_BYTES = 1_000_000

# 这些目录通常包含依赖、版本控制数据或缓存，含有敏感信息，不应作为创建文件的目录。
BLOCKED_DIRECTORY_NAMES = {".git", ".venv", "venv", "__pycache__"}


class CreateMarkdownInput(BaseModel):
	"""创建 Markdown 文件的结构化输入。"""

	file_path: str = Field(
		description="项目根目录下的相对文件路径，必须以 .md 结尾，例如 notes/summary.md。"
	)
	content: str = Field(description="要写入 Markdown 文件的 UTF-8 文本内容。")

	@model_validator(mode="after")
	def validate_file_path(self) -> "CreateMarkdownInput":
		"""确保文件路径非空、相对且指向 Markdown 文件。"""
		if not self.file_path.strip():
			raise ValueError("file_path 不能为空")

		requested_path = Path(self.file_path)
		
		if requested_path.is_absolute():
			raise ValueError("请提供项目根目录下的相对路径")
		if requested_path.suffix.lower() != ".md":
			raise ValueError("只允许创建 .md 文件")
		return self


def create_markdown_tool(request: CreateMarkdownInput) -> dict[str, str | int]:
	"""在项目内创建 UTF-8 Markdown 文件；若文件已存在则拒绝覆盖。"""
	
	content_size = len(request.content.encode("utf-8"))
	if content_size > MAX_FILE_SIZE_BYTES:
		raise ValueError(
			f"内容超过 {MAX_FILE_SIZE_BYTES} 字节的创建上限：{request.file_path}"
		)

	requested_path = Path(request.file_path)
	resolved_path = (PROJECT_ROOT / requested_path).resolve()
	
	try:
		resolved_path.relative_to(PROJECT_ROOT)
	except ValueError as error:
		raise ValueError("不允许在项目目录以外创建文件") from error

	if BLOCKED_DIRECTORY_NAMES.intersection(resolved_path.parts):
		raise ValueError("不允许在依赖、缓存或版本控制目录中创建文件")
	if not resolved_path.parent.is_dir():
		raise FileNotFoundError(f"目标目录不存在：{resolved_path.parent}")

	# 独占创建可避免检查与写入之间的竞态，也不会覆盖已有文件。
	try:
		with resolved_path.open("x", encoding="utf-8", newline="") as file:
			file.write(request.content)
	except FileExistsError as error:
		raise FileExistsError(
			f"文件已存在，不会覆盖：{request.file_path}"
		) from error

	return {
		"status": "created",
		"file_path": resolved_path.relative_to(PROJECT_ROOT).as_posix(),
		"bytes_written": content_size,
	}
