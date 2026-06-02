import os
import glob
import logging
import threading
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from .config import EMBEDDING_LLM
from .chunking import chunk_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
CHROMA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'datas')
COLLECTION_NAME = "deep_research"
CHUNK_SIZE = 512
MULTIPASS_MERGE_SIZE = 4
_CHROMA_INSTANCE = None
_LOCK = threading.Lock()

def _get_collection():
    global _CHROMA_INSTANCE

    if _CHROMA_INSTANCE is None:
        with _LOCK:
            if _CHROMA_INSTANCE is None:
                _CHROMA_INSTANCE = Chroma(
                    collection_name=COLLECTION_NAME,
                    embedding_function=EMBEDDING_LLM,
                    persist_directory=CHROMA_DIR
                )
    return _CHROMA_INSTANCE

def build_index(folder: str = DATA_DIR):
    """构建向量索引（基于句子级切块 + multipass 合并）。

    切块策略变更（相对旧版 RecursiveCharacterTextSplitter）：
      - 旧: chunk_size=512, chunk_overlap=32, 递归字符分割
      - 新: 句子级切分 + multipass 多通道合并，保证语义完整性
    """
    documents = []
    collection = _get_collection()

    # 旧策略已被替换:
    #   text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
    # 新策略: sentence_level_chunk() + multipass_merge(), 见 chunking.py

    files_paths = glob.glob(os.path.join(folder, "*.md"))
    if not files_paths:
        logger.warning("未找到任何 .md 文件: %s", folder)
        return 0

    total_chunks = 0

    for file_path in files_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # ---- 新切块策略 ----
        chunks = chunk_document(
            content,
            chunk_size=CHUNK_SIZE,
            enable_multipass=True,
            multipass_merge_size=MULTIPASS_MERGE_SIZE,
        )

        filename = os.path.basename(file_path)
        logger.info("%s: 共 %d 个块（multipass=%d）", filename, len(chunks), MULTIPASS_MERGE_SIZE)

        for idx, chunk_text in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source": filename,
                        "chunk_index": idx,
                        "chunk_size": len(chunk_text),
                        "method": "sentence+multipass",
                    }
                )
            )
            total_chunks += 1

    if not documents:
        return 0

    collection.add_documents(documents)
    logger.info("索引完成: %d 个文档, %d 个块", len(files_paths), total_chunks)

    return total_chunks

def search(query: str, k: int = 3):
    collection = _get_collection()

    results = collection.similarity_search_with_score(
        query=query,
        k=k
    )

    outputs = []
    for doc, score in results:
        if score < 1.3:
            outputs.append(
                {
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                    "chunk_index": doc.metadata.get("chunk_index", -1),
                    "distance": score
                }
            )
    
    print("----------------")
    print(f"查询的关键词是：{query}\n")
    for key, value in enumerate(outputs):
        text = outputs[key]["text"][:30]
        score = outputs[key]["distance"]
        print(f"查询到第{key}条，相关性是{score}，内容是:{text}")

    return outputs

# build_index()
# results = search("你好")
# print(results)