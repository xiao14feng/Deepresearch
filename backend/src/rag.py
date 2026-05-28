import os
import glob
import threading
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from .config import EMBEDDING_LLM

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
CHROMA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'datas')
COLLECTION_NAME = "deep_research"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 32
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

def build_index(floder: str = DATA_DIR):
    documents = []
    collection = _get_collection()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    files_paths = glob.glob(os.path.join(floder, "*.md"))
    if not files_paths:
        return 0

    for file_path in files_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = text_splitter.split_text(content)
        filename = os.path.basename(file_path)

        for idx, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "chunk_index": idx,
                    }
                )
            )

    if not documents:
        return 0

    collection.add_documents(documents)

    return len(documents)

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