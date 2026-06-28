"""
LLM Planner Agent

Creates autonomous data science
execution plans using an LLM.
"""

from config.llm import ask_llm
from prompts.prompts import PLANNER_PROMPT


class PlannerAgent:
    """
    AI Planner Agent

    Responsible for creating
    the execution workflow.
    """

    def __init__(self):

        self.current_task = None
        self.current_plan = None

    def create_plan(self, task: str):

        self.current_task = task

        prompt = PLANNER_PROMPT.format(
            task=task
        )

        try:

            response = ask_llm(
                prompt
            )

            self.current_plan = response

            return response

        except Exception as e:

            return (
                f"Planner Agent Error: "
                f"{str(e)}"
            )

    def get_plan_steps(self):

        if self.current_plan is None:
            return []

        steps = []

        for line in self.current_plan.split("\n"):

            line = line.strip()

            if line:

                if line[0].isdigit():
                    steps.append(line)

        return steps


planner = PlannerAgent()