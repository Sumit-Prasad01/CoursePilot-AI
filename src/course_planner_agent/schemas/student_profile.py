from typing import List, Dict, Optional
from pydantic import BaseModel


class StudentProfile(BaseModel):
    """
    Structured student profile extracted from query
    """
    completed_courses: List[str] = []
    grades: Dict[str, str] = {}
    target_program: Optional[str] = None
    max_credits: Optional[int] = None
    raw_query: Optional[str] = None