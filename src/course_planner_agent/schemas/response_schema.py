from typing import List, Optional
from pydantic import BaseModel


class ResponseSchema(BaseModel):
    """
    Structured response format for the assistant
    """
    answer: Optional[str] = None
    why: Optional[str] = None
    citations: List[str] = []
    clarifying_questions: List[str] = []
    assumptions: Optional[str] = None
    error: Optional[str] = None