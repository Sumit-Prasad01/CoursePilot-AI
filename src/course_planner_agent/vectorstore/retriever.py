from langchain_community.vectorstores import FAISS
from typing import List
from langchain_core.documents import Document
from src.course_planner_agent.utils.logger import logger


def get_retriever(vectorstore: FAISS, k: int = 5):
    """
    Returns a configured retriever
    """
    try:
        logger.info(f"Initializing retriever with top-k={k}")
        return vectorstore.as_retriever(search_kwargs={"k": k})

    except Exception as e:
        logger.error(f"Error creating retriever: {e}")
        raise


def retrieve_documents(retriever, query: str) -> List[Document]:
    """
    Retrieve relevant documents for a query
    """
    try:
        logger.info(f"Retrieving documents for query: {query}")

        docs = retriever.invoke(query)

        logger.info(f"Retrieved {len(docs)} documents")
        return docs

    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise