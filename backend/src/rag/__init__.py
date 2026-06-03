"""RAG 核心模块 — 切块、索引、检索、评估。"""

from .chunking import (
    count_tokens,
    split_sentences,
    sentence_level_chunk,
    multipass_merge,
    chunk_document,
    generate_mini_chunks,
)
from .indexing import (
    build_index,
    _get_collection,
    simple_search,
    CHROMA_DIR,
    DATA_DIR,
    COLLECTION_NAME,
)
from .retrieval import (
    retrieve,
    hybrid_search,
    expand_query,
    rrf_fuse,
    llm_filter,
    BM25,
)
from .evaluation import (
    evaluate_retrieval,
    evaluate_answer,
)

__all__ = [
    "count_tokens",
    "split_sentences",
    "sentence_level_chunk",
    "multipass_merge",
    "chunk_document",
    "generate_mini_chunks",
    "build_index",
    "_get_collection",
    "simple_search",
    "retrieve",
    "hybrid_search",
    "expand_query",
    "rrf_fuse",
    "llm_filter",
    "BM25",
    "evaluate_retrieval",
    "evaluate_answer",
    "CHROMA_DIR",
    "DATA_DIR",
    "COLLECTION_NAME",
]
