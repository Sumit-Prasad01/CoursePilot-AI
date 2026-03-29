from typing import Dict, Any
from src.course_planner_agent.state.state import GraphState
from src.course_planner_agent.utils.logger import logger
from src.course_planner_agent.schemas.student_profile import StudentProfile

def intake_node(state: GraphState) -> GraphState:
    """
    Extracts basic student info from query.
    (For now: pass-through + placeholder parsing)
    """
    try:
        logger.info("Running Intake Node")

        query = state.get("query", "")

        # Minimal parsing (can be upgraded later with LLM)
        student_profile = StudentProfile(
            raw_query=query,
            completed_courses=courses,
            grades={},
            target_program=None,
            max_credits=None
        )

        # Very basic heuristic extraction (optional improvement later)
        # Example: detect course codes
        import re
        courses = re.findall(r"CS\d{3}", query)
        if courses:
            student_profile["completed_courses"] = courses

        state["student_profile"] = student_profile.model_dump()

        logger.info(f"Extracted student profile: {student_profile}")

        return state

    except Exception as e:
        logger.error(f"Intake Node failed: {e}")
        state["error"] = str(e)
        return state