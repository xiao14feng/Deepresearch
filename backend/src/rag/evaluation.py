"""
RAG 评估模块（参考 Onyx RAGAS 评估体系）

直接运行：python -m src.rag.evaluation
或：PYTHONIOENCODING=utf-8 ../.venv/Scripts/python -m src.rag.evaluation

指标：
1. Faithfulness — 事实一致性，答案是否忠实于上下文
2. Context Recall — 上下文召回，检索内容是否覆盖问题所需
3. Context Precision — 上下文精度，检索内容是否准确
4. Answer Relevancy — 答案相关性，答案是否切题

使用 LLM 作为评判（无需额外依赖）。
"""

import json
import logging

from ..config import LLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 评估 Prompt
# ---------------------------------------------------------------------------

_PROMPT_FACTUALNESS = """
你是一个 RAG 评估专家。请判断以下"答案"是否忠实于给出的"上下文"。

判断标准：
- 如果答案中的所有陈述都能从上下文中找到依据 → 得分 1.0
- 如果答案中的部分陈述无法从上下文中找到依据 → 得分 0.5
- 如果答案与上下文完全矛盾或凭空捏造 → 得分 0.0

请以 JSON 格式输出：
{{ "score": 0.0~1.0, "reason": "简要说明" }}

上下文：{context}
答案：{answer}
"""

_PROMPT_CONTEXT_RECALL = """
你是一个 RAG 评估专家。请判断"检索到的上下文"是否覆盖了"问题"所需的关键信息。

判断标准：
- 1.0 = 上下文完整覆盖了回答问题所需的所有信息
- 0.5 = 上下文覆盖了部分信息，但有一些信息缺失
- 0.0 = 上下文几乎没有回答问题的相关信息

请以 JSON 格式输出：
{{ "score": 0.0~1.0, "reason": "简要说明" }}

问题：{query}
上下文：{context}
"""

_PROMPT_ANSWER_RELEVANCY = """
你是一个 RAG 评估专家。请判断"答案"与"问题"的相关程度。

判断标准：
- 1.0 = 答案直接回答了问题，完全相关
- 0.5 = 答案部分相关，但没有完全回答问题
- 0.0 = 答案与问题完全不相关

请以 JSON 格式输出：
{{ "score": 0.0~1.0, "reason": "简要说明" }}

问题：{query}
答案：{answer}
"""


# ---------------------------------------------------------------------------
# LLM 评估辅助
# ---------------------------------------------------------------------------

def _llm_judge(prompt: str) -> tuple[float, str]:
    """调用 LLM 进行评判，返回 (score, reason)。"""
    try:
        content = LLM.invoke(prompt).content
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        parsed = json.loads(content)
        score = float(parsed.get("score", 0.0))
        reason = parsed.get("reason", "")
        return max(0.0, min(1.0, score)), reason
    except Exception as e:
        logger.warning("LLM 评判失败: %s", e)
        return 0.0, str(e)


# ---------------------------------------------------------------------------
# 公开评估接口
# ---------------------------------------------------------------------------

def evaluate_retrieval(query: str, contexts: list[str]) -> dict:
    """评估检索质量。

    Args:
        query: 用户问题。
        contexts: 检索到的上下文文本列表。

    Returns:
        {context_recall: float, context_precision: float, details: [...]}
    """
    if not contexts:
        return {"context_recall": 0.0, "context_precision": 0.0, "details": []}

    # 合并上下文用于评估
    merged_context = "\n---\n".join(contexts)

    # Context Recall：检索到的上下文是否覆盖了问题所需的信息
    recall_score, recall_reason = _llm_judge(
        _PROMPT_CONTEXT_RECALL.format(query=query, context=merged_context[:3000])
    )

    # Context Precision：每个上下文片段的精确度
    precision_scores = []
    for ctx in contexts:
        p_score, _ = _llm_judge(
            _PROMPT_CONTEXT_RECALL.format(query=query, context=ctx[:1500])
        )
        precision_scores.append(p_score)

    context_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0

    result = {
        "context_recall": round(recall_score, 4),
        "context_precision": round(context_precision, 4),
        "details": [
            {"fragment": i, "precision": round(s, 4)}
            for i, s in enumerate(precision_scores)
        ],
    }

    logger.info("检索评估: recall=%.2f, precision=%.2f", recall_score, context_precision)
    return result


def evaluate_answer(query: str, answer: str, contexts: list[str]) -> dict:
    """评估最终答案质量。

    Args:
        query: 用户问题。
        answer: 生成的答案。
        contexts: 检索到的上下文文本列表。

    Returns:
        {faithfulness, answer_relevancy, ...}
    """
    merged_context = "\n---\n".join(contexts) if contexts else "无上下文"

    # Faithfulness
    faithfulness, faith_reason = _llm_judge(
        _PROMPT_FACTUALNESS.format(context=merged_context[:3000], answer=answer[:2000])
    )

    # Answer Relevancy
    relevancy, rel_reason = _llm_judge(
        _PROMPT_ANSWER_RELEVANCY.format(query=query, answer=answer[:2000])
    )

    result = {
        "faithfulness": round(faithfulness, 4),
        "answer_relevancy": round(relevancy, 4),
        "faithfulness_reason": faith_reason,
        "relevancy_reason": rel_reason,
    }

    logger.info("答案评估: faithfulness=%.2f, relevancy=%.2f", faithfulness, relevancy)
    return result


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys
    _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.setLevel(logging.INFO)

    print("=" * 50)
    print("RAG 评估工具")
    print("=" * 50)

    # 从命令行参数读取
    args = _sys.argv[1:]

    if len(args) >= 1:
        query = args[0]
    else:
        query = input("请输入查询问题: ").strip()

    if len(args) >= 2:
        context_file = args[1]
        with open(context_file, "r", encoding="utf-8") as f:
            contexts = [line.strip() for line in f if line.strip()]
    else:
        # 从检索管道自动获取
        from .indexing import build_index
        from .retrieval import retrieve
        build_index()
        results = retrieve(query, k=3, enable_llm_filter=False)
        contexts = [r["text"] for r in results]
        if not contexts:
            print("\n检索为空，请手动输入上下文（空行结束）:")
            contexts = []
            while True:
                line = input()
                if not line:
                    break
                contexts.append(line)

    if len(args) >= 3:
        answer = args[2]
    else:
        try:
            gen = input("是否生成答案？（y/n，默认 n）: ").strip().lower()
        except EOFError:
            gen = "n"
        if gen == "y":
            from ..config import LLM
            ctx_text = "\n".join(contexts[:3])
            answer = LLM.invoke(f"请基于以下内容回答问题：\n{ctx_text}\n\n问题：{query}").content
        else:
            answer = ""

    # 执行评估
    print(f"\n查询: {query}\n")
    print("--- 检索评估 ---")
    ret = evaluate_retrieval(query, contexts)
    print(f"  Context Recall:    {ret['context_recall']:.2f}")
    print(f"  Context Precision: {ret['context_precision']:.2f}")

    if answer:
        print("\n--- 答案评估 ---")
        ans = evaluate_answer(query, answer, contexts)
        print(f"  Faithfulness:      {ans['faithfulness']:.2f}")
        print(f"  Answer Relevancy:  {ans['answer_relevancy']:.2f}")
        print(f"  事实理由: {ans.get('faithfulness_reason', '')[:100]}")
        print(f"  相关理由: {ans.get('relevancy_reason', '')[:100]}")

    print("\n评估完成")
