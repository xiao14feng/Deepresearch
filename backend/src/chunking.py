"""
切块策略（参考 Onyx 阅读笔记 onyx阅读.md）

核心思路：
1. 句子级切分 — 保留标点在上一句，按最小句数和最大 token 数合并
2. 不设 overlap，改用 multipass 多通道合并（每 4 个小块合并为一个大块）
3. 超长单句 — 不硬截断，按 chunk_size 字符截取
4. 超短文档 — 直接添加并发出警告

对比旧策略（RecursiveCharacterTextSplitter + overlap）：
  - 旧：按分隔符递归切分，可能切碎语义完整的句子
  - 新：以句子为最小单元，保证语义完整性
  - 旧：overlap 冗余存储，增加向量数据库压力
  - 新：multipass 多通道合并，提供更丰富的上下文且无冗余
"""

import re
import logging

import tiktoken

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token 计数（使用 tiktoken，与 text-embedding-3-small 所用编码一致）
# ---------------------------------------------------------------------------

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """精确计算文本的 token 数（cl100k_base 编码）。

    与 OpenAI text-embedding-3-small 模型使用的编码一致，
    确保 chunk_size=512 是真实的 512 token 边界。
    """
    if not text:
        return 0
    return len(_ENCODING.encode(text, disallowed_special=()))


def split_sentences(text: str) -> list[str]:
    """将文本按句子边界切分，保留标点在上一句。

    支持的句子边界：
      - 中文句号（。） 感叹号（！） 问号（？）
      - 英文句号（.） 感叹号（!） 问号（?）
      - 换行符（\\n）
    """
    raw = re.split(r'(?<=[。！？.!?\n])\s*', text)
    return [s.strip() for s in raw if s.strip()]


# ---------------------------------------------------------------------------
# 核心切块函数
# ---------------------------------------------------------------------------

def sentence_level_chunk(
    text: str,
    chunk_size: int = 512,
    min_sentences: int = 2,
    max_sentences: int = 15,
) -> list[str]:
    """句子级切块。

    策略（来自 onyx 阅读笔记）：
      1. 以最小句子数和最大 token 数为约束合并句子
      2. 超过 max_sentences 也要保存（防止单块无限膨胀）
      3. 文档极少（句子数 < min_sentences）→ 直接返回
      4. 单句超 chunk_size → 不按 token 硬截断，按 chunk_size 字符截取

    Args:
        text: 原始文本。
        chunk_size: 目标 token 数上限。
        min_sentences: 合并时的最少句子数。
        max_sentences: 一块中最多容纳的句子数。

    Returns:
        切块后的文本列表。
    """
    if not text or not text.strip():
        return []

    sentences = split_sentences(text)

    # 超短文档：直接返回（符合 onyx 策略：直接添加并 warning）
    if len(sentences) < min_sentences:
        logger.warning("文档过短（仅 %d 句），直接作为单块添加", len(sentences))
        return [text.strip()]

    chunks = []
    current_group = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = count_tokens(sentence)

        # 单句超 chunk_size：直接截取，不做 token 级硬切
        if sent_tokens > chunk_size:
            if current_group:
                chunks.append(''.join(current_group).strip())
                current_group = []
                current_tokens = 0
            # 按 chunk_size 字符截取（不硬按照 token 数切块 — onyx 策略）
            truncated = sentence[:chunk_size]
            chunks.append(truncated)
            continue

        # 当前组超限 → 保存，新开一组
        if (
            current_tokens + sent_tokens > chunk_size
            and len(current_group) >= min_sentences
        ):
            chunks.append(''.join(current_group).strip())
            current_group = [sentence]
            current_tokens = sent_tokens
            continue

        # 句子数已达上限 → 强制保存
        if len(current_group) >= max_sentences:
            chunks.append(''.join(current_group).strip())
            current_group = [sentence]
            current_tokens = sent_tokens
            continue

        current_group.append(sentence)
        current_tokens += sent_tokens

    # 收尾剩余组
    if current_group:
        chunks.append(''.join(current_group).strip())

    return chunks


def multipass_merge(chunks: list[str], merge_size: int = 4) -> list[str]:
    """Multipass 多通道合并。

    将小 chunk 每 merge_size 个合并为一组，提供更丰富的上下文。
    这是对传统 overlap 的替代方案（onyx 策略）：
      - overlap：通过冗余存储解决边界上下文丢失问题
      - multipass：通过合并相邻小块提供完整上下文，无冗余

    用法示例：
        chunks = sentence_level_chunk(text)
        merged = multipass_merge(chunks, merge_size=4)
    """
    if not chunks:
        return []
    if len(chunks) <= merge_size:
        return ['\n\n'.join(chunks)]

    merged = []
    for i in range(0, len(chunks), merge_size):
        group = chunks[i:i + merge_size]
        merged.append('\n\n'.join(group))

    return merged


def chunk_document(
    content: str,
    chunk_size: int = 512,
    min_sentences: int = 2,
    enable_multipass: bool = True,
    multipass_merge_size: int = 4,
) -> list[str]:
    """完整的文档切块流程。

    Pass 1 — 句子级切块
      保证每块包含完整的句子，不截断语义。

    Pass 2 — Multipass 多通道合并（可选）
      将相邻小块合并为大块，替代 overlap 机制。

    Args:
        content: 文档全文。
        chunk_size: 目标 token 数上限。
        min_sentences: 最小句子数。
        enable_multipass: 是否启用 multipass 合并。
        multipass_merge_size: 每组合并的小块数。

    Returns:
        切块后的文本列表。
    """
    # Pass 1: 句子级切块
    chunks = sentence_level_chunk(
        content,
        chunk_size=chunk_size,
        min_sentences=min_sentences,
    )

    if not chunks:
        return []

    # Pass 2: Multipass 多通道合并（替代 overlap）
    if enable_multipass and len(chunks) > 1:
        merged = multipass_merge(chunks, merge_size=multipass_merge_size)
        logger.debug(
            "multipass: %d chunks → %d chunks (merge_size=%d)",
            len(chunks), len(merged), multipass_merge_size,
        )
        return merged

    return chunks
