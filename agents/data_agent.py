"""
Data Agent

Analyzes the structure of datasets.
"""

import pandas as pd


class DataAgent:

    def analyze(self, df: pd.DataFrame):

        analysis = {

            "rows": df.shape[0],

            "columns": df.shape[1],

            "column_names": list(df.columns),

            "dtypes":
                df.dtypes.astype(str).to_dict(),

            "missing_values":
                df.isnull().sum().to_dict(),

            "numerical_columns":
                list(
                    df.select_dtypes(
                        include=["number"]
                    ).columns
                ),

            "categorical_columns":
                list(
                    df.select_dtypes(
                        include=["object","string"]
                    ).columns
                )
        }

        return analysis


data_agent = DataAgent()