import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from src.rag import search, build_index
from src.config import LLM

HERE = os.path.dirname(os.path.abspath(__file__))


def score_retrieval(query: str, keywords: list[str], docs: list[dict]) -> dict:
    docs_text = ""
    for doc in docs:
        docs_text += f"\n-【来源:{doc['source']}, 索引:{doc['chunk_index']}】\n 内容:{doc['text'][:300]}"

    prompt = f"""
查询: {query}
期望覆盖的关键词: {', '.join(keywords)}
检索到的文档:{docs_text}
请严格返回JSON格式，包含:
  - relevance: 1-5，检索结果整体和查询的相关性
  - hit_keywords: 命中了哪些期望关键词（数组）
  - comment: 一句话评价（中文，20字以内）
只返回JSON，不要其他内容。
"""

    result = LLM.invoke(prompt)
    try:
        content = result.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]
        parsed = json.loads(content)
        # 防 LLM 返回列表或非 dict 的情况
        if not isinstance(parsed, dict):
            return {"relevance": 0, "hit_keywords": [], "comment": f"LLM返回了{type(parsed).__name__}而非dict"}
        return parsed
    except json.JSONDecodeError:
        return {"relevance": 0, "hit_keywords": [], "comment": "解析失败"}



def main():
    build_index()

    with open(os.path.join(HERE, "test_database.json"), "r", encoding="utf-8") as f:
        queries = json.load(f)

    total = {"relevance": 0, "keyword_hits": 0}
    results = []

    for i, q in enumerate(queries):
        query = q["query"]
        keywords = q["expected_keywords"]
        print(f"[{i+1}/{len(queries)}] {query}")

        docs = search(query, k=3)

        score = score_retrieval(query, keywords, docs)

        hit_rate = len(score["hit_keywords"]) / len(keywords) if keywords else 0

        results.append({
            "query": query,
            "relevance": score["relevance"],
            "hit_rate": round(hit_rate, 2),
            "hit_keywords": score["hit_keywords"],
            "total_keywords": keywords,
            "top_sources": [d["source"] for d in docs],
            "comment": score["comment"]
        })

        total["relevance"] += score["relevance"]
        total["keyword_hits"] += hit_rate
        print(f"  相关性: {score['relevance']}/5 | 关键词命中: {hit_rate:.0%} | {score['comment']}")

    n = len(results)
    avg_rel = total["relevance"] / n
    avg_hit = total["keyword_hits"] / n

    summary = {
        "total_queries": n,
        "avg_relevance": round(avg_rel, 2),
        "avg_keyword_hit_rate": round(avg_hit, 2),
        "score_label": "良好" if avg_rel >= 3.5 else "一般" if avg_rel >= 2.5 else "较差",
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "details": results
    }

    output_path = os.path.join(HERE, "results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*40}")
    print(f"RAG 评测完成")
    print(f"查询数: {n}")
    print(f"平均相关性: {avg_rel:.1f}/5")
    print(f"平均关键词命中率: {avg_hit:.0%}")
    print(f"总体评价: {summary['score_label']}")
    print(f"结果: {output_path}")


if __name__ == "__main__":
    main()
