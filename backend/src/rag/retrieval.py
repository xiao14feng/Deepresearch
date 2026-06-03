"""
高级检索管道（参考 Onyx 阅读笔记 onyx阅读.md）

核心优化（相对旧版单向量搜索）：
1. BM25 关键词检索 + 向量检索混合搜索
2. 查询扩展 — 语义改写 + 关键词提取
3. 多路并行检索 — 不同查询/不同 alpha 权重
4. 加权 RRF 融合 — 合并多路结果
5. LLM 过滤 — 剔除无关结果
6. 上下文扩展 — 对保留结果添加上下更多内容
"""

import re
import math
import json
import logging
from collections import Counter
from typing import Optional

from ..config import LLM

from .indexing import _get_collection, _CHROMA_INSTANCE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BM25 实现（无外部依赖）
# ---------------------------------------------------------------------------

_BM25_INSTANCE: Optional["BM25"] = None
_CHUNK_CORPUS: list[str] = []


class BM25:
    """轻量 BM25 关键词检索。

    BM25 公式：score(D,Q) = Σ IDF(q_i) * (f(q_i,D)*(k1+1)) / (f(q_i,D) + k1*(1-b+b*|D|/avgdl))
    """

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.avgdl = sum(len(tokens) for tokens in self.tokenized_corpus) / max(len(corpus), 1)

        # 预计算 IDF
        self.idf: dict[str, float] = {}
        n_docs = len(corpus)
        if n_docs > 0:
            df: Counter = Counter()
            for tokens in self.tokenized_corpus:
                df.update(set(tokens))
            for term, doc_freq in df.items():
                self.idf[term] = math.log(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """中文按字符切分，英文按空格/标点切分。"""
        text = text.lower()
        tokens = []
        for token in re.findall(r'[一-鿿]|[a-z]+', text):
            if token:
                tokens.append(token)
        return tokens

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """返回 (corpus_index, score) 列表，按 BM25 得分降序。"""
        query_tokens = self._tokenize(query)
        if not query_tokens or not self.corpus:
            return []

        scores = []
        for idx, tokens in enumerate(self.tokenized_corpus):
            score = 0.0
            doc_len = len(tokens)
            for term in set(query_tokens):
                if term not in self.idf:
                    continue
                tf = tokens.count(term)
                idf = self.idf[term]
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# ---------------------------------------------------------------------------
# 查询扩展
# ---------------------------------------------------------------------------

_QUERY_EXPAND_PROMPT = """
你是一个查询扩展助手。给定用户的一个搜索查询，请生成以下变体：

1. **语义改写**：保持原意但补充完整语义，适合向量检索
2. **关键词版本**：提取核心关键词，适合 BM25 关键词检索

请以 JSON 格式输出，不要包含其他内容：
{{
    "semantic": "语义改写后的查询",
    "keyword": "关键词1 关键词2 关键词3"
}}
原查询：{query}
"""


def expand_query(query: str) -> dict[str, str]:
    """用 LLM 扩展查询，返回 {'original': ..., 'semantic': ..., 'keyword': ...}。"""
    result = {"original": query, "semantic": query, "keyword": query}
    try:
        content = LLM.invoke(_QUERY_EXPAND_PROMPT.format(query=query)).content
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        parsed = json.loads(content)
        if isinstance(parsed, dict):
            result["semantic"] = parsed.get("semantic", query)
            result["keyword"] = parsed.get("keyword", query)
            logger.debug("查询扩展: original=%s, semantic=%s, keyword=%s", query, result["semantic"], result["keyword"])
    except Exception as e:
        logger.warning("查询扩展失败: %s，回退使用原始查询", e)

    return result


# ---------------------------------------------------------------------------
# 混合检索（BM25 + 向量）
# ---------------------------------------------------------------------------

def _lazy_init_bm25():
    """惰性初始化 BM25 索引。"""
    global _BM25_INSTANCE, _CHUNK_CORPUS

    if _BM25_INSTANCE is None:
        try:
            collection = _get_collection()
            all_docs = collection.get(include=["documents"])
            documents = all_docs.get("documents", []) or []
            if documents:
                _CHUNK_CORPUS = documents
                _BM25_INSTANCE = BM25(documents)
                logger.info("BM25 索引已构建，共 %d 个文档", len(documents))
        except Exception as e:
            logger.warning("BM25 索引构建失败: %s", e)


def hybrid_search(
    query: str,
    alpha: float = 0.5,
    k: int = 5,
) -> list[dict]:
    """混合检索：score = alpha * 向量得分 + (1 - alpha) * BM25 得分。

    alpha 控制向量和 BM25 的混合权重：
      - alpha=1.0 纯向量检索
      - alpha=0.0 纯 BM25 检索
      - alpha=0.5 均衡混合

    Returns:
        [{text, source, chunk_index, vector_score, bm25_score, hybrid_score}, ...]
    """
    _lazy_init_bm25()

    # 1. 向量检索
    vector_results = []
    if _CHROMA_INSTANCE is not None or True:
        try:
            collection = _get_collection()
            raw = collection.similarity_search_with_score(query=query, k=k * 3)
            for doc, score in raw:
                vector_results.append({
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                    "chunk_index": doc.metadata.get("parent_idx", -1),
                    "vector_score": score,
                })
        except Exception as e:
            logger.warning("向量检索失败: %s", e)

    # 2. BM25 检索
    bm25_results = []
    if _BM25_INSTANCE:
        try:
            raw = _BM25_INSTANCE.search(query, k=k * 3)
            for idx, score in raw:
                bm25_results.append({
                    "text": _CHUNK_CORPUS[idx],
                    "chunk_index": idx,
                    "bm25_score": score,
                })
        except Exception as e:
            logger.warning("BM25 检索失败: %s", e)

    # 3. 合并结果（用 text 作为 key 去重）
    seen_texts: dict[str, dict] = {}

    max_v_score = 1.0
    if vector_results:
        max_v_score = max(r["vector_score"] for r in vector_results) or 1.0
    for r in vector_results:
        r["bm25_score"] = 0.0
        r["normalized_vector"] = 1.0 - r["vector_score"] / max_v_score
        seen_texts[r["text"]] = r

    max_b_score = 1.0
    if bm25_results:
        max_b_score = max(r["bm25_score"] for r in bm25_results) or 1.0
    for r in bm25_results:
        text = r["text"]
        if text in seen_texts:
            seen_texts[text]["bm25_score"] = r["bm25_score"] / max_b_score
        else:
            r["normalized_vector"] = 0.0
            r["source"] = ""
            r["vector_score"] = 99.0
            r["bm25_score"] = r["bm25_score"] / max_b_score
            seen_texts[text] = r

    # 4. 计算混合得分并排序
    results = []
    for text, r in seen_texts.items():
        hybrid = alpha * r["normalized_vector"] + (1 - alpha) * r["bm25_score"]
        results.append({
            "text": r.get("text", text),
            "source": r.get("source", ""),
            "chunk_index": r.get("chunk_index", -1),
            "vector_score": r.get("vector_score", 99.0),
            "bm25_score": r.get("bm25_score", 0.0),
            "hybrid_score": round(hybrid, 4),
        })

    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return results[:k]


# ---------------------------------------------------------------------------
# RRF 融合
# ---------------------------------------------------------------------------

def rrf_fuse(
    ranked_lists: list[list[dict]],
    weights: Optional[list[float]] = None,
    k: int = 60,
    top_n: int = 5,
) -> list[dict]:
    """加权 RRF 融合多条排序列表。

    RRF 得分 = Σ(weight_i / (k + rank_i))

    Args:
        ranked_lists: 多个排序结果列表，每个列表元素须包含 'text' 字段
        weights: 每个列表的权重，None 则均等
        k: RRF 常数
        top_n: 返回前 N 条

    Returns:
        按 RRF 得分降序排列的融合结果
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    text_scores: dict[str, float] = {}
    text_items: dict[str, dict] = {}

    for rank_list, weight in zip(ranked_lists, weights):
        for rank, item in enumerate(rank_list, start=1):
            text = item.get("text", "")
            if not text:
                continue
            rrf_score = weight / (k + rank)
            text_scores[text] = text_scores.get(text, 0) + rrf_score
            if text not in text_items:
                text_items[text] = item

    sorted_texts = sorted(text_scores, key=text_scores.get, reverse=True)

    results = []
    for text in sorted_texts[:top_n]:
        item = dict(text_items[text])
        item["rrf_score"] = round(text_scores[text], 4)
        results.append(item)

    return results


# ---------------------------------------------------------------------------
# LLM 过滤
# ---------------------------------------------------------------------------

_LLM_FILTER_PROMPT = """
你是一个检索结果相关性判断专家。你的任务是从检索结果中筛选出与用户查询真正相关的内容。

判断标准：
- 保留：内容与查询主题直接相关，即使只有部分相关也保留
- 剔除：内容与查询完全无关

请以 JSON 数组格式输出你决定保留的索引编号：
[0, 2, 3]

只输出 JSON，不要其他内容。

用户查询：{query}
检索结果：
{results}
"""


def llm_filter(query: str, results: list[dict]) -> list[dict]:
    """用 LLM 筛选检索结果，保留相关结果。"""
    if not results:
        return results

    lines = []
    for i, r in enumerate(results):
        snippet = r.get("text", "")[:150]
        lines.append(f"[{i}] {snippet}")
    result_summary = "\n".join(lines)

    try:
        content = LLM.invoke(_LLM_FILTER_PROMPT.format(query=query, results=result_summary)).content
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        keep_indices = json.loads(content)
        if isinstance(keep_indices, list) and all(isinstance(i, int) for i in keep_indices):
            filtered = [results[i] for i in keep_indices if 0 <= i < len(results)]
            logger.debug("LLM 过滤: %d → %d 条", len(results), len(filtered))
            return filtered
    except Exception as e:
        logger.warning("LLM 过滤失败: %s，保留全部结果", e)

    return results


# ---------------------------------------------------------------------------
# 完整检索管道
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    k: int = 5,
    enable_query_expansion: bool = True,
    enable_llm_filter: bool = True,
    alpha_semantic: float = 0.5,
    alpha_keyword: float = 0.2,
    weight_original: float = 0.5,
    weight_semantic: float = 1.3,
    weight_keyword: float = 1.0,
) -> list[dict]:
    """完整的高级检索管道。

    流程：
    1. 查询扩展 — 生成语义改写和关键词版本
    2. 多路并行检索 — 每个查询变体按不同 alpha 混合检索
    3. 加权 RRF 融合 — 融合所有检索结果
    4. (可选) LLM 过滤 — 剔除无关结果

    Args:
        query: 用户原始查询
        k: 最终返回结果数
        enable_query_expansion: 是否启用查询扩展
        enable_llm_filter: 是否启用 LLM 过滤
        alpha_semantic: 语义查询的混合权重（向量得分比重）
        alpha_keyword: 关键词查询的混合权重
        weight_original: 原始查询的 RRF 权重
        weight_semantic: 语义改写查询的 RRF 权重
        weight_keyword: 关键词查询的 RRF 权重

    Returns:
        [{text, source, chunk_index, hybrid_score/rrf_score, ...}, ...]
    """
    logger.info("检索管道开始 — query=%s", query)

    # Step 1: 查询扩展
    queries = [query]
    weights = [weight_original]
    alphas = [alpha_semantic]

    if enable_query_expansion:
        expansions = expand_query(query)
        queries.append(expansions["semantic"])
        weights.append(weight_semantic)
        alphas.append(alpha_semantic)
        queries.append(expansions["keyword"])
        weights.append(weight_keyword)
        alphas.append(alpha_keyword)
        logger.debug("多路查询: %s", [q[:20] for q in queries])

    # Step 2: 多路并行检索
    all_ranked = []
    for q, a in zip(queries, alphas):
        try:
            results = hybrid_search(q, alpha=a, k=k * 2)
            if results:
                all_ranked.append(results)
                logger.debug("  查询=%s alpha=%.1f → %d 结果", q[:20], a, len(results))
        except Exception as e:
            logger.warning("查询失败 (%s): %s", q[:20], e)

    if not all_ranked:
        logger.warning("所有检索路径均失败，返回空结果")
        return []

    # Step 3: RRF 融合
    fused = rrf_fuse(all_ranked, weights=weights, top_n=k * 2)
    logger.debug("RRF 融合: %d 路 → %d 条", len(all_ranked), len(fused))

    # Step 4: LLM 过滤
    if enable_llm_filter and fused:
        fused = llm_filter(query, fused)
        logger.debug("LLM 过滤后: %d 条", len(fused))

    return fused[:k]
