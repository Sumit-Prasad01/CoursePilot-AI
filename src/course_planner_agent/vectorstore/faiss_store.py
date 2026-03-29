from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from src.course_planner_agent.utils.logger import logger


class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]):
        return self.model.encode(texts, show_progress_bar=True)

    def embed_query(self, text: str):
        return self.model.encode([text])[0]


def build_faiss_index(documents: List[Document], save_path: str):
    """
    Create and save FAISS index
    """
    try:
        logger.info("Building FAISS index...")

        embedding_model = EmbeddingModel()

        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embedding_model
        )

        vectorstore.save_local(save_path)

        logger.info(f"FAISS index saved at {save_path}")

    except Exception as e:
        logger.error(f"Error building FAISS index: {e}")
        raise


def load_faiss_index(load_path: str) -> FAISS:
    """
    Load FAISS index
    """
    try:
        logger.info(f"Loading FAISS index from {load_path}")

        embedding_model = EmbeddingModel()

        vectorstore = FAISS.load_local(
            load_path,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )

        return vectorstore

    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        raise