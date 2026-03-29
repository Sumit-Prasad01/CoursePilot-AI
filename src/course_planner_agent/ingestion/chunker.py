from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.course_planner_agent.utils.logger import logger


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks with metadata preserved
    """
    try:
        logger.info("Starting document chunking...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "Document", "Prerequisites", "Credits"]
        )

        chunks = splitter.split_documents(documents)

        # Add simple chunk id metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i

        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    except Exception as e:
        logger.error(f"Error during chunking: {e}")
        raise