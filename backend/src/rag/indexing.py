"""
向量索引构建与管理（Chroma + text-embedding-3-small）

功能：
1. 连接 / 初始化 Chroma 集合（单例模式）
2. 读取 Markdown 文档 → 切块 → 写入 Chroma
3. 简化检索（simple_search，兼容旧接口）
4. Mini-chunk 也一并写入（metadata 标记）
"""

import os
import glob
import logging
import threading
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ..config import EMBEDDING_LLM
from .chunking import chunk_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env'))
CHROMA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'chroma')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'datas')
COLLECTION_NAME = "deep_research"

_CHROMA_INSTANCE = None
_LOCK = threading.Lock()


def _get_collection():
    """获取 Chroma 集合（单例）。"""
    global _CHROMA_INSTANCE

    if _CHROMA_INSTANCE is None:
        with _LOCK:
            if _CHROMA_INSTANCE is None:
                _CHROMA_INSTANCE = Chroma(
                    collection_name=COLLECTION_NAME,
                    embedding_function=EMBEDDING_LLM,
                    persist_directory=CHROMA_DIR,
                )
    return _CHROMA_INSTANCE


def build_index(folder: str = None) -> int:
    """构建向量索引（句子级切块 + multipass 合并 + mini-chunk）。

    切块策略变更（相对旧版 RecursiveCharacterTextSplitter）：
      - 旧: chunk_size=512, chunk_overlap=32, 递归字符分割
      - 新: 句子级切分 + multipass 多通道合并 + mini-chunk 精细检索

    Args:
        folder: 文档目录，默认 DATA_DIR。

    Returns:
        写入的文档块总数。
    """
    if folder is None:
        folder = DATA_DIR

    documents = []
    collection = _get_collection()

    files_paths = glob.glob(os.path.join(folder, "*.md"))
    if not files_paths:
        logger.warning("未找到任何 .md 文件: %s", folder)
        return 0

    total_chunks = 0

    for file_path in files_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 切块（含 mini-chunk）
        chunks = chunk_document(content, enable_mini_chunks=True)

        filename = os.path.basename(file_path)
        logger.info(
            "%s: 共 %d 个块（含 %d 个 mini-chunk）",
            filename,
            len(chunks),
            sum(1 for c in chunks if c["is_mini_chunk"]),
        )

        for chunk_info in chunks:
            documents.append(
                Document(
                    page_content=chunk_info["text"],
                    metadata={
                        "source": filename,
                        "is_mini_chunk": chunk_info["is_mini_chunk"],
                        "parent_idx": chunk_info["parent_idx"],
                        "method": "sentence+multipass+minichunk",
                    },
                )
            )
            total_chunks += 1

    if not documents:
        return 0

    # Chroma.add_documents 自动去重（如果 ids 相同会覆盖）
    # 这里不指定 ids，每次重建索引会追加
    collection.add_documents(documents)
    logger.info("索引完成: %d 个文档, %d 个块", len(files_paths), total_chunks)

    return total_chunks


def simple_search(query: str, k: int = 3) -> list[dict]:
    """简化向量检索（兼容旧接口，仅向量搜索 + 距离阈值过滤）。

    Args:
        query: 查询文本。
        k: 返回结果数。

    Returns:
        [{text, source, chunk_index, distance}, ...]
    """
    collection = _get_collection()
    results = collection.similarity_search_with_score(query=query, k=k)

    outputs = []
    for doc, score in results:
        outputs.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source", ""),
            "chunk_index": doc.metadata.get("parent_idx", -1),
            "distance": score,
        })

    logger.info("简单检索: query=%s → %d 条结果", query[:30], len(outputs))
    return outputs
