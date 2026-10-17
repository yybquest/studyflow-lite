import sys

from tools.pdf_to_text import PdfToTextInput, pdf_to_text_tool

if __name__ == "__main__":
    """用法：python pdf2md.py article/xxx.pdf"""
    print(pdf_to_text_tool(PdfToTextInput(file_path=sys.argv[1])))