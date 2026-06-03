"""
新旧切块策略对比评测

比较维度：
  - 索引统计：块数、平均块大小、token 利用率
  - 检索质量：相关性评分、关键词命中率
  - 定性对比：同一查询的检索结果差异

用法：
  python test/compare_chunking.py
"""

import os
import json
import sys
import shutil
import re
from dataclasses import dataclass, field, asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import tiktoken
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import EMBEDDING_LLM
from src.chunking import chunk_document as new_chunking

_ENCODING = tiktoken.get_encoding("cl100k_base")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "datas")
COMPARE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "compare")
OLD_DIR = os.path.join(COMPARE_DIR, "old_chroma")
NEW_DIR = os.path.join(COMPARE_DIR, "new_chroma")
COLLECTION_NAME = "compare"
TEST_QUERIES_PATH = os.path.join(os.path.dirname(__file__), "test_database.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "compare_results.json")
TOP_K = 3

# ---------------------------------------------------------------------------
# 旧策略（RecursiveCharacterTextSplitter + overlap）
# ---------------------------------------------------------------------------
def old_chunking(content: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=32,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", "，", ",", " ", ""],
    )
    return splitter.split_text(content)


# ---------------------------------------------------------------------------
# 索引构建
# ---------------------------------------------------------------------------
@dataclass
class IndexStats:
    strategy: str = ""
    total_docs: int = 0
    total_chunks: int = 0
    avg_chunk_chars: float = 0.0
    avg_chunk_tokens: float = 0.0
    token_utilization: float = 0.0  # 平均 token / chunk_size
    chunk_sizes_tokens: list[int] = field(default_factory=list)

def build_index(strategy: str, chunk_fn, persist_dir: str) -> IndexStats:
    """建索引并返回统计"""
    # 清空
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    collection = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=EMBEDDING_LLM,
        persist_directory=persist_dir,
    )

    stats = IndexStats(strategy=strategy)
    documents = []
    file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.md")))

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = chunk_fn(content)
        filename = os.path.basename(file_path)

        for idx, chunk_text in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={"source": filename, "chunk_index": idx},
                )
            )
            t = len(_ENCODING.encode(chunk_text, disallowed_special=()))
            stats.chunk_sizes_tokens.append(t)

    stats.total_docs = len(file_paths)
    stats.total_chunks = len(documents)

    if stats.chunk_sizes_tokens:
        stats.avg_chunk_chars = sum(len(d.page_content) for d in documents) / len(documents)
        stats.avg_chunk_tokens = sum(stats.chunk_sizes_tokens) / len(stats.chunk_sizes_tokens)
        stats.token_utilization = stats.avg_chunk_tokens / 512 * 100

    if documents:
        collection.add_documents(documents)

    return stats


# ---------------------------------------------------------------------------
# 检索评测
# ---------------------------------------------------------------------------
@dataclass
class QueryResult:
    query: str = ""
    expected_keywords: list[str] = field(default_factory=list)
    old_relevance: int = 0
    new_relevance: int = 0
    old_hit_rate: float = 0.0
    new_hit_rate: float = 0.0
    old_hit_keywords: list[str] = field(default_factory=list)
    new_hit_keywords: list[str] = field(default_factory=list)
    old_sources: list[str] = field(default_factory=list)
    new_sources: list[str] = field(default_factory=list)
    old_texts: list[str] = field(default_factory=list)
    new_texts: list[str] = field(default_factory=list)
    old_comment: str = ""
    new_comment: str = ""
    winner: str = ""  # "old", "new", "tie"


def score_retrieval(query: str, keywords: list[str], docs: list[dict]) -> dict:
    """调用 LLM 评估检索质量"""
    docs_text = ""
    for doc in docs:
        docs_text += f"\n-【来源:{doc['source']}, 索引:{doc['chunk_index']}】\n 内容:{doc['text'][:300]}"

    prompt = f"""查询: {query}
期望覆盖的关键词: {', '.join(keywords)}
检索到的文档:{docs_text}
请严格返回JSON格式，包含:
  - relevance: 1-5，检索结果整体和查询的相关性
  - hit_keywords: 命中了哪些期望关键词（数组）
  - comment: 一句话评价（中文，20字以内）
只返回JSON，不要其他内容。"""

    from src.config import LLM
    result = LLM.invoke(prompt)
    try:
        content = result.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return {"relevance": 0, "hit_keywords": [], "comment": f"返回了{type(parsed).__name__}"}
        return parsed
    except json.JSONDecodeError:
        return {"relevance": 0, "hit_keywords": [], "comment": "解析失败"}


def search_collection(collection, query: str, k: int = TOP_K):
    """检索并格式化为统一结构"""
    results = collection.similarity_search_with_score(query, k=k)
    outputs = []
    for doc, score in results:
        outputs.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source", ""),
            "chunk_index": doc.metadata.get("chunk_index", -1),
            "distance": score,
        })
    return outputs


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
import glob


def main():
    print("=" * 60)
    print("  新旧切块策略对比评测")
    print("=" * 60)

    # ---- 1. 建索引 ----
    print("\n▶ 构建旧策略索引（RecursiveCharacterTextSplitter）...")
    old_stats = build_index("旧策略", old_chunking, OLD_DIR)
    print(f"   块数: {old_stats.total_chunks}, 平均 {old_stats.avg_chunk_tokens:.0f} tokens/块")
    print(f"   Token 利用率: {old_stats.token_utilization:.1f}%")

    print("\n▶ 构建新策略索引（句子级 + Multipass）...")
    new_stats = build_index("新策略", new_chunking, NEW_DIR)
    print(f"   块数: {new_stats.total_chunks}, 平均 {new_stats.avg_chunk_tokens:.0f} tokens/块")
    print(f"   Token 利用率: {new_stats.token_utilization:.1f}%")

    # ---- 2. 统计对比 ----
    print("\n" + "=" * 60)
    print("  索引统计对比")
    print("=" * 60)
    print(f"  {'指标':<20} {'旧策略':>12} {'新策略':>12} {'变化':>12}")
    print(f"  {'-'*56}")
    changes = []
    for label, old_val, new_val, fmt in [
        ("文档数", old_stats.total_docs, new_stats.total_docs, "{:>12d}"),
        ("总块数", old_stats.total_chunks, new_stats.total_chunks, "{:>12d}"),
        ("平均字符/块", old_stats.avg_chunk_chars, new_stats.avg_chunk_chars, "{:>12.0f}"),
        ("平均 tokens/块", old_stats.avg_chunk_tokens, new_stats.avg_chunk_tokens, "{:>12.0f}"),
        ("Token 利用率", old_stats.token_utilization, new_stats.token_utilization, "{:>11.1f}%"),
    ]:
        diff = new_val - old_val
        diff_str = f"{diff:+.0f}" if isinstance(diff, (int, float)) and not isinstance(diff, bool) else ""
        if "利用率" in label:
            diff_str = f"{diff:+.1f}%"
        print(f"  {label:<20} {fmt.format(old_val)} {fmt.format(new_val)} {diff_str:>12}")

    # ---- 3. 检索评测 ----
    print("\n" + "=" * 60)
    print("  检索质量评测")
    print("=" * 60)

    old_collection = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=EMBEDDING_LLM,
        persist_directory=OLD_DIR,
    )
    new_collection = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=EMBEDDING_LLM,
        persist_directory=NEW_DIR,
    )

    with open(TEST_QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    results = []
    total_old_rel = 0
    total_new_rel = 0
    total_old_hit = 0.0
    total_new_hit = 0.0

    for i, q in enumerate(queries):
        query = q["query"]
        keywords = q["expected_keywords"]
        print(f"\n  [{i+1}/{len(queries)}] {query}")

        # 旧策略检索
        old_docs = search_collection(old_collection, query)
        old_score = score_retrieval(query, keywords, old_docs)
        old_hit_rate = len(old_score["hit_keywords"]) / len(keywords) if keywords else 0

        # 新策略检索
        new_docs = search_collection(new_collection, query)
        new_score = score_retrieval(query, keywords, new_docs)
        new_hit_rate = len(new_score["hit_keywords"]) / len(keywords) if keywords else 0

        # 判定胜者
        old_total = old_score["relevance"] + old_hit_rate
        new_total = new_score["relevance"] + new_hit_rate
        if new_total > old_total:
            winner = "new"
        elif old_total > new_total:
            winner = "old"
        else:
            winner = "tie"

        total_old_rel += old_score["relevance"]
        total_new_rel += new_score["relevance"]
        total_old_hit += old_hit_rate
        total_new_hit += new_hit_rate

        r = QueryResult(
            query=query,
            expected_keywords=keywords,
            old_relevance=old_score["relevance"],
            new_relevance=new_score["relevance"],
            old_hit_rate=round(old_hit_rate, 2),
            new_hit_rate=round(new_hit_rate, 2),
            old_hit_keywords=old_score["hit_keywords"],
            new_hit_keywords=new_score["hit_keywords"],
            old_sources=[d["source"] for d in old_docs],
            new_sources=[d["source"] for d in new_docs],
            old_texts=[d["text"][:100] for d in old_docs],
            new_texts=[d["text"][:100] for d in new_docs],
            old_comment=old_score.get("comment", ""),
            new_comment=new_score.get("comment", ""),
            winner=winner,
        )
        results.append(r)

        # 打印对比
        hit_icon = "↑" if new_hit_rate > old_hit_rate else "↓" if new_hit_rate < old_hit_rate else "="
        rel_icon = "↑" if new_score["relevance"] > old_score["relevance"] else "↓" if new_score["relevance"] < old_score["relevance"] else "="
        print(f"    旧→ rel:{old_score['relevance']}/5 hit:{old_hit_rate:.0%}")
        print(f"    新→ rel:{new_score['relevance']}/5 {rel_icon}  hit:{new_hit_rate:.0%} {hit_icon}")
        if old_score["comment"] or new_score["comment"]:
            print(f"    旧: {old_score['comment']}")
            print(f"    新: {new_score['comment']}")

    # ---- 4. 汇总 ----
    n = len(results)
    print("\n" + "=" * 60)
    print("  汇总")
    print("=" * 60)
    avg_old_rel = total_old_rel / n
    avg_new_rel = total_new_rel / n
    avg_old_hit = total_old_hit / n
    avg_new_hit = total_new_hit / n

    print(f"  {'指标':<20} {'旧策略':>10} {'新策略':>10} {'变化':>10}")
    print(f"  {'-'*50}")
    print(f"  {'平均相关性':<20} {avg_old_rel:>8.1f}/5 {avg_new_rel:>8.1f}/5 {avg_new_rel-avg_old_rel:>+9.1f}")
    print(f"  {'平均关键词命中率':<20} {avg_old_hit:>9.0%} {avg_new_hit:>9.0%} {avg_new_hit-avg_old_hit:>+9.0%}")

    # 胜负统计
    wins = {"new": 0, "old": 0, "tie": 0}
    for r in results:
        wins[r.winner] += 1
    print(f"  {'单题胜出':<20} {'':>10} {'':>10}")
    print(f"    {'新策略胜':<16} {wins['new']:>2d}/{n}")
    print(f"    {'旧策略胜':<16} {wins['old']:>2d}/{n}")
    print(f"    {'平局':<16} {wins['tie']:>2d}/{n}")

    # ---- 5. 定性对比展示 ----
    print("\n" + "=" * 60)
    print("  定性对比——检索结果差异")
    print("=" * 60)
    shown = 0
    for r in results:
        if r.winner != "tie" and shown < 3:
            shown += 1
            label = "新策略更好" if r.winner == "new" else "旧策略更好"
            print(f"\n  ▶ {label}: {r.query}")
            print(f"  ┌─ 旧策略 TOP1 ────────────────────────────────")
            print(f"  │ {r.old_texts[0][:80] if r.old_texts else '(无)'}")
            print(f"  ├─ 新策略 TOP1 ────────────────────────────────")
            print(f"  │ {r.new_texts[0][:80] if r.new_texts else '(无)'}")
            print(f"  └────────────────────────────────────────────")

    # ---- 6. 保存结果 ----
    output = {
        "index_stats": {
            "old": {k: v for k, v in asdict(old_stats).items() if k != "chunk_sizes_tokens"},
            "new": {k: v for k, v in asdict(new_stats).items() if k != "chunk_sizes_tokens"},
        },
        "summary": {
            "avg_relevance_old": round(avg_old_rel, 2),
            "avg_relevance_new": round(avg_new_rel, 2),
            "avg_hit_rate_old": round(avg_old_hit, 2),
            "avg_hit_rate_new": round(avg_new_hit, 2),
            "new_wins": wins["new"],
            "old_wins": wins["old"],
            "ties": wins["tie"],
        },
        "details": [
            {k: v for k, v in asdict(r).items() if k not in ("old_texts", "new_texts")}
            for r in results
        ],
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  完整结果已保存: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
