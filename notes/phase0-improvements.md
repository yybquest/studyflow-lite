# 阶段 0（文档摄取）改进清单 — 暂未实施

> 文档摄取层：PDF 转文本 + 扩展文件类型。以下按实用价值排序，都不急。

## 第一优先：语料里已经踩到的坑

1. **Word 文档（.docx）**：`notes/cv.doc` 就是。`.docx` 用 `python-docx`；`.doc` 老格式需 LibreOffice 转。
2. **.epub**：`article/` 里就有 `AI-Agents-in-Depth-zh-CN.epub`。epub 是 zip 包 HTML，解包抽文本即可。
3. **非 UTF-8 编码**：现在只按 UTF-8 读，GBK 中文文件会崩。加编码探测/回退（charset-normalizer）。
4. **Jupyter Notebook（.ipynb）**：本质 JSON，解析 code/markdown cell 即可，性价比极高。

## 第二优先：PDF 抽取质量（能读但不精）

5. **表格**：复杂三线表会乱，换 pdfplumber/camelot 或转 markdown 表格保留结构。
6. **公式（LaTeX）**：目前当普通文本，结构丢失。
7. **图表**：目前完全跳过。可抽 caption 做检索线索，或接视觉模型生成描述。
8. **扫描版 PDF（OCR）**：无文本层需 tesseract/paddleocr。

## 第三优先：更广格式覆盖

9. **PPT / Excel（.pptx / .xlsx）**：python-pptx / openpyxl。
10. **HTML**：网页文章、文档导出。
11. **.tex**：LaTeX 源文件。

## 第四优先：工程性细节

12. **大文件分页读取**：pdf_to_text 支持「只抽第 1~10 页」。
13. **增量处理**：记录已转换的 PDF，跳过重复抽取。
14. **图片 OCR**：截图、图片里的文字。
