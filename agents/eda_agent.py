"""
EDA Agent

Performs exploratory data analysis.
"""

import pandas as pd


class EDAAgent:

    def analyze(self, df: pd.DataFrame):

        report = {

            "shape": df.shape,

            "missing_values":
                df.isnull().sum().to_dict(),

            "duplicates":
                int(df.duplicated().sum()),

            "summary":
                df.describe(
                    include="all"
                ).to_dict()
        }

        return report


eda_agent = EDAAgent()