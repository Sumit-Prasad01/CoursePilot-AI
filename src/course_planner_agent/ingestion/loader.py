from langchain_community.document_loaders import PyPDFLoader
from typing import List
from langchain_core.documents import Document
from src.course_planner_agent.utils.logger import logger


def load_pdf(file_path: str) -> List[Document]:
    """
    Load PDF and return LangChain Document objects
    """
    try:
        logger.info(f"Loading PDF from {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages")
        return documents

    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        raise