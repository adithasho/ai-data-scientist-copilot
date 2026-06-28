import pandas as pd


class RouterAgent:

    def route(
        self,
        df
    ):

        target = (
            df.columns[-1]
        )

        y = df[target]

        # String target
        if (
            y.dtype ==
            "object"
        ):

            return {

                "task":
                    "classification",

                "agent":
                    "classification_agent"
            }

        # Binary classification
        if (
            y.nunique() <= 2
        ):

            return {

                "task":
                    "classification",

                "agent":
                    "classification_agent"
            }

        # Regression
        return {

            "task":
                "regression",

            "agent":
                "regression_agent"
        }


router = RouterAgent()