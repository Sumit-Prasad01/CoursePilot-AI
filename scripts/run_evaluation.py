from dotenv import load_dotenv

from evaluation.evaluator import evaluate
from src.course_planner_agent.utils.logger import logger


load_dotenv()


def main():
    try:
        logger.info("Running evaluation script...")
        evaluate()
        logger.info("Evaluation finished successfully")

    except Exception as e:
        logger.error(f"Evaluation script failed: {e}")


if __name__ == "__main__":
    main()