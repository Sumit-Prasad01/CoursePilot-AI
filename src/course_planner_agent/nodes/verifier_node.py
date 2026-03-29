from groq import Groq
import os
import re
import json
from src.course_planner_agent.state.state import GraphState
from src.course_planner_agent.utils.logger import logger
from src.course_planner_agent.utils.prompt_loader import load_prompt
from src.course_planner_agent.schemas.response_schema import ResponseSchema


VERIFIER_PROMPT_PATH = "src/course_planner_agent/prompts/verifier_prompt.txt"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def verifier_node(state: GraphState) -> GraphState:
    """
    LLM-based verification with structured output
    """
    try:
        logger.info("Running Verifier Node")

        answer = state.get("answer", {})
        answer_text = json.dumps(answer, indent=2)

        if not answer:
            state["error"] = "No answer generated"
            return state

        # Load verifier prompt
        verifier_prompt = load_prompt(VERIFIER_PROMPT_PATH)

        # Create verification input
        verification_input = f"""
Response to verify:

{answer_text}
"""

        # LLM verification
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": verifier_prompt},
                {"role": "user", "content": verification_input}
            ],
            temperature=0
        )

        final_text = response.choices[0].message.content

        # Extract citations
        citations = list(set(re.findall(r"\[Chunk\s*\d+\]", final_text)))

        # Wrap into structured schema
        response_obj = ResponseSchema(
            answer=final_text,
            citations=citations,
            clarifying_questions=[],
            assumptions=None,
            error=None
        )

        # Store structured output
        state["citations"] = citations
        state["final_output"] = response_obj.dict()

        logger.info(f"Verifier completed with {len(citations)} citations")

        return state

    except Exception as e:
        logger.error(f"Verifier Node failed: {e}")

        error_response = ResponseSchema(
            answer=None,
            citations=[],
            clarifying_questions=[],
            assumptions=None,
            error=str(e)
        )

        state["final_output"] = error_response.dict()
        state["error"] = str(e)

        return state