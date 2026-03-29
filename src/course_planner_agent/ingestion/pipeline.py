from typing import List
from langchain_core.documents import Document
from src.course_planner_agent.ingestion.loader import load_pdf
from src.course_planner_agent.ingestion.chunker import chunk_documents
from src.course_planner_agent.utils.logger import logger


def run_ingestion_pipeline(file_path: str) -> List[Document]:
    """
    Full ingestion pipeline:
    Load PDF → Chunk → Return processed documents
    """
    try:
        logger.info("Starting ingestion pipeline...")

        documents = load_pdf(file_path)
        chunks = chunk_documents(documents)

        logger.info("Ingestion pipeline completed")
        return chunks

    except Exception as e:
        logger.error(f"Ingestion pipeline failed: {e}")
        raise