import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from langgraph.types import Send
from mcp.server import FastMCP
from src.state import ResearchState
from src.prompts import PROMPT_PLAN
from src.config import LLM
from src.tools import search_web
from src.rag.indexing import simple_search, build_index
from src.rag.retrieval import retrieve

server = FastMCP("search_database")

@server.tool()
def search_rag(title: str, query: str) -> dict:
    # 使用高级检索管道：混合搜索 + 查询扩展 + RRF 融合 + LLM 过滤
    outputs = retrieve(query, k=5, enable_llm_filter=True)
    # 回退到简单向量检索
    if not outputs:
        outputs = simple_search(query, k=3)

    texts = []
    scores = []
    for value in outputs:
        text = value.get("text", "")
        score = value.get("hybrid_score", value.get("rrf_score", value.get("distance", 0)))
        scores.append(score)
        texts.append(f"检索到的文档{len(texts)}, {text}\n")

    rag_results = {
        "title": title,
        "score": list(scores),
        "output": list(texts)
    }

    return rag_results

@server.tool()
def search_web1(title: str, query: str) -> dict:
    web_results = []
    texts = []
    outputs, error = search_web(query)

    for key, value in enumerate(outputs):
        title = outputs[key]["title"]
        snippet = outputs[key]["snippet"]
        texts.append(f"标题:{title}, 内容:{snippet}\n")
    
    web_results = {
        "title": title,
        "output": list(texts)
    }

    return web_results

if __name__ == "__main__":
    server.run(transport="stdio")