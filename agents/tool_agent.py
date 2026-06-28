"""
Tool Agent V2

Executes analytical tools
selected by the Reasoning Agent.
"""

import pandas as pd


class ToolAgent:

    def execute(
        self,
        action,
        question,
        df
    ):

        question = question.lower()

        # =========================
        # AGGREGATION
        # =========================

        if action == "aggregation":

            for col in df.columns:

                if col.lower() in question:

                    try:

                        return (
                            df[col]
                            .value_counts()
                            .to_dict()
                        )

                    except:
                        pass

            return {

                "rows":
                    len(df)
            }

        # =========================
        # FILTER
        # =========================

        if action == "filter":

            if "churn" in question:

                for col in df.columns:

                    if (
                        "churn"
                        in col.lower()
                    ):

                        return (
                            df[
                                df[col]
                                ==
                                "Yes"
                            ]
                            .head()
                        )

            return (
                df.head()
            )

        # =========================
        # STATISTICS
        # =========================

        if action == "statistics":

            return (

                df
                .describe(
                    include="all"
                )
                .to_dict()
            )

        # =========================
        # MISSING VALUES
        # =========================

        if action == "missing":

            return (

                df
                .isnull()
                .sum()
                .to_dict()
            )

        # =========================
        # UNIQUE VALUES
        # =========================

        if action == "unique":

            result = {}

            for col in df.columns:

                if (

                    df[col]
                    .dtype
                    ==
                    "object"
                ):

                    result[col] = (

                        df[col]
                        .unique()
                        .tolist()
                    )

            return result

        # =========================
        # CORRELATION
        # =========================

        if action == "correlation":

            numeric = (

                df
                .select_dtypes(
                    include="number"
                )
            )

            return (

                numeric
                .corr()
                .round(2)
                .to_dict()
            )

        # =========================
        # SORT
        # =========================

        if action == "sort":

            numeric = (

                df
                .select_dtypes(
                    include="number"
                )
                .columns
            )

            if len(numeric):

                return (

                    df
                    .sort_values(

                        numeric[0],

                        ascending=False
                    )
                    .head()
                )

        # =========================
        # SHAPE
        # =========================

        if action == "shape":

            return {

                "rows":
                    len(df),

                "columns":
                    len(df.columns)
            }

        # =========================
        # DESCRIBE
        # =========================

        if action == "describe":

            return (

                df
                .describe(
                    include="all"
                )
                .to_dict()
            )

        # =========================
        # DEFAULT
        # =========================

        return {

            "message":
                "No tool available."
        }


tool_agent = ToolAgent()