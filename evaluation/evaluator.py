import json
from typing import Dict, List
from dotenv import load_dotenv

from src.course_planner_agent.graph.workflow import CoursePlannerWorkflow
from src.course_planner_agent.utils.logger import logger


load_dotenv()


def load_test_queries(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(text: str) -> str:
    return (text or "").lower()


def check_keywords(text: str, keywords: List[str]) -> bool:
    text = normalize(text)
    return all(k.lower() in text for k in keywords)


def check_not_keywords(text: str, keywords: List[str]) -> bool:
    text = normalize(text)
    return all(k.lower() not in text for k in keywords)


def check_decision(text: str, expected: str) -> bool:
    text = normalize(text)

    if expected == "eligible":
        return "eligible" in text

    if expected == "not eligible":
        return "not eligible" in text or "not eligible" in text

    if expected == "abstain":
        return "i don’t have" in text or "i don't have" in text

    return False


def evaluate():
    try:
        logger.info("Starting evaluation with ground truth...")

        workflow = CoursePlannerWorkflow()
        test_cases = load_test_queries("evaluation/test_queries.json")

        total = len(test_cases)
        correct = 0

        results = []

        for test in test_cases:
            query = test["query"]
            expected = test.get("expected", {})

            logger.info(f"Evaluating: {query}")

            result = workflow.run(query)
            output = result.get("final_output", {})

            answer_text = ""
            if isinstance(output, dict):
                answer_text = output.get("answer", "")
            else:
                answer_text = str(output)

            # ---- checks ----
            decision_ok = True
            include_ok = True
            exclude_ok = True

            if "decision" in expected:
                decision_ok = check_decision(answer_text, expected["decision"])

            if "must_include" in expected:
                include_ok = check_keywords(answer_text, expected["must_include"])

            if "must_not_include" in expected:
                exclude_ok = check_not_keywords(answer_text, expected["must_not_include"])

            is_correct = decision_ok and include_ok and exclude_ok

            if is_correct:
                correct += 1

            results.append({
                "query": query,
                "expected": expected,
                "correct": is_correct,
                "output": answer_text
            })

        accuracy = (correct / total) * 100

        logger.info("\n" + "=" * 60)
        logger.info("🎯 FINAL EVALUATION (GROUND TRUTH)")
        logger.info("=" * 60)
        logger.info(f"Total: {total}")
        logger.info(f"Correct: {correct}")
        logger.info(f"Accuracy: {accuracy:.2f}%")

        with open("evaluation/final_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info("Ground truth evaluation completed")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")


if __name__ == "__main__":
    evaluate()