"""
Reasoning Agent V2

Determines how the AI should
respond to a user query.
"""


class ReasoningAgent:

    def think(
        self,
        question: str
    ):

        question = (
            question
            .lower()
            .strip()
        )

        # ==================================
        # AGGREGATION
        # ==================================

        if any(
            word in question
            for word in [

                "how many",

                "count",

                "number of"
            ]
        ):

            return {

                "action":
                    "aggregation",

                "tool":
                    "pandas"
            }

        # ==================================
        # FILTERING
        # ==================================

        if any(
            word in question
            for word in [

                "show",

                "find",

                "list",

                "display"
            ]
        ):

            return {

                "action":
                    "filter",

                "tool":
                    "pandas"
            }

        # ==================================
        # STATISTICS
        # ==================================

        if any(
            word in question
            for word in [

                "average",

                "mean",

                "median",

                "statistics",

                "std",

                "minimum",

                "maximum"
            ]
        ):

            return {

                "action":
                    "statistics",

                "tool":
                    "pandas"
            }

        # ==================================
        # DESCRIBE DATASET
        # ==================================

        if any(
            word in question
            for word in [

                "describe",

                "summary",

                "summarize"
            ]
        ):

            return {

                "action":
                    "describe",

                "tool":
                    "pandas"
            }

        # ==================================
        # MISSING VALUES
        # ==================================

        if any(
            word in question
            for word in [

                "missing",

                "null",

                "nan"
            ]
        ):

            return {

                "action":
                    "missing",

                "tool":
                    "pandas"
            }

        # ==================================
        # CORRELATION
        # ==================================

        if any(
            word in question
            for word in [

                "correlation",

                "relationship",

                "correlated"
            ]
        ):

            return {

                "action":
                    "correlation",

                "tool":
                    "pandas"
            }

        # ==================================
        # UNIQUE VALUES
        # ==================================

        if any(
            word in question
            for word in [

                "unique",

                "categories",

                "distinct"
            ]
        ):

            return {

                "action":
                    "unique",

                "tool":
                    "pandas"
            }

        # ==================================
        # SORTING
        # ==================================

        if any(
            word in question
            for word in [

                "highest",

                "largest",

                "top",

                "lowest",

                "smallest"
            ]
        ):

            return {

                "action":
                    "sort",

                "tool":
                    "pandas"
            }

        # ==================================
        # DATASET SHAPE
        # ==================================

        if any(
            word in question
            for word in [

                "rows",

                "columns",

                "shape",

                "size"
            ]
        ):

            return {

                "action":
                    "shape",

                "tool":
                    "pandas"
            }

        # ==================================
        # VISUALIZATION
        # ==================================

        if any(
            word in question
            for word in [

                "plot",

                "graph",

                "chart",

                "visualize",

                "histogram",

                "scatter",

                "bar chart",

                "heatmap"
            ]
        ):

            return {

                "action":
                    "visualization",

                "tool":
                    "matplotlib"
            }

        # ==================================
        # MODEL QUESTIONS
        # ==================================

        if any(
            word in question
            for word in [

                "model",

                "accuracy",

                "feature",

                "importance",

                "xgboost",

                "random forest",

                "logistic",

                "decision tree",

                "why",

                "improve"
            ]
        ):

            return {

                "action":
                    "model_reasoning",

                "tool":
                    "llm"
            }

        # ==================================
        # EDA QUESTIONS
        # ==================================

        if any(
            word in question
            for word in [

                "duplicate",

                "outlier",

                "distribution",

                "eda"
            ]
        ):

            return {

                "action":
                    "eda_reasoning",

                "tool":
                    "llm"
            }

        # ==================================
        # BUSINESS QUESTIONS
        # ==================================

        if any(
            word in question
            for word in [

                "business",

                "recommendation",

                "insight",

                "strategy"
            ]
        ):

            return {

                "action":
                    "business_reasoning",

                "tool":
                    "llm"
            }

        # ==================================
        # DEFAULT
        # ==================================

        return {

            "action":
                "general_reasoning",

            "tool":
                "llm"
        }


reasoning_agent = (
    ReasoningAgent()
)