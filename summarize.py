import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from tools.summarizer import summarize_paper

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
print(summarize_paper(client, sys.argv[1]))