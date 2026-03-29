import sys
from dotenv import load_dotenv

from src.course_planner_agent.graph.workflow import CoursePlannerWorkflow
from src.course_planner_agent.utils.logger import logger


load_dotenv()


def main():
    try:
        logger.info("Starting Course Planner")

        workflow = CoursePlannerWorkflow()

        # Get query from CLI or input
        if len(sys.argv) > 1:
            query = " ".join(sys.argv[1:])
        else:
            query = input("Enter your query: ")

        result = workflow.run(query)

        logger.info("\n" + "=" * 50)
        logger.info("FINAL OUTPUT:\n")
        logger.info(result["final_output"]["answer"])
        logger.info("\n" + "=" * 50)

        if result.get("error"):
            print("\nERROR:", result["error"])

    except Exception as e:
        logger.error(f"Run failed: {e}")


if __name__ == "__main__":
    main()