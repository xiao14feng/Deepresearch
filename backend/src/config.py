import os
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
LLM = ChatOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    model=os.getenv("LLM_MODEL")
)
EMBEDDING_LLM = OpenAIEmbeddings(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    model="text-embedding-3-small",
)