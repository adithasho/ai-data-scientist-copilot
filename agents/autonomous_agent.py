"""
Autonomous AI Agent

Master orchestrator
for all AI agents.
"""

from agents.reasoning_agent import reasoning_agent
from agents.tool_agent import tool_agent
from agents.explanation_agent import explanation_agent


class AutonomousAgent:

    def run(
        self,
        question,
        df
    ):

        # =====================
        # REASON
        # =====================

        decision = (
            reasoning_agent
            .think(question)
        )

        # =====================
        # ACT
        # =====================

        result = (
            tool_agent
            .execute(

                decision[
                    "action"
                ],

                question,

                df
            )
        )

        # =====================
        # EXPLAIN
        # =====================

        explanation = (
            explanation_agent
            .explain(

                question,

                result
            )
        )

        return {

            "decision":
                decision,

            "result":
                result,

            "answer":
                explanation
        }


autonomous_agent = (
    AutonomousAgent()
)