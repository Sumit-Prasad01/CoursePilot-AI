import json
from groq import Groq
import os

from src.course_planner_agent.state.state import GraphState
from src.course_planner_agent.utils.logger import logger
from src.course_planner_agent.utils.prompt_loader import load_prompt


SYSTEM_PROMPT_PATH = "src/course_planner_agent/prompts/system_prompt.txt"
PLANNER_PROMPT_PATH = "src/course_planner_agent/prompts/planner_prompt.txt"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def format_docs(docs) -> str:
    formatted = []
    for doc in docs:
        content = doc.page_content
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        formatted.append(f"[Chunk {chunk_id}]\n{content}")
    return "\n\n".join(formatted)


def safe_json_load(text: str):
    """
    Safely parse JSON from LLM output
    """
    try:
        return json.loads(text)
    except:
        # try to extract JSON substring
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError("Invalid JSON output from model")


def planner_node(state: GraphState) -> GraphState:
    try:
        logger.info("Running Planner Node")

        query = state.get("query", "")
        docs = state.get("retrieved_docs", [])

        context = format_docs(docs)

        system_prompt = load_prompt(SYSTEM_PROMPT_PATH)
        planner_template = load_prompt(PLANNER_PROMPT_PATH)

        prompt = planner_template.format(
            context=context,
            query=query
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        raw_output = response.choices[0].message.content

        parsed = safe_json_load(raw_output)

        # store structured output directly
        state["answer"] = parsed

        logger.info("Planner Node completed (JSON mode)")

        return state

    except Exception as e:
        logger.error(f"Planner Node failed: {e}")
        state["error"] = str(e)
        return state