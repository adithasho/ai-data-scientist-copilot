""""
CSV Loader Tool

Loads datasets for the
AI Data Scientist Copilot.
"""

import pandas as pd


class CSVLoader:

    def __init__(self):
        self.df = None

    def load(self, filepath: str):

        try:
            self.df = pd.read_csv(filepath)

            print(
                f"Dataset loaded successfully "
                f"({self.df.shape[0]} rows, "
                f"{self.df.shape[1]} columns)"
            )

            return self.df

        except Exception as e:
            print(
                f"Error loading dataset: {e}"
            )

            return None


csv_loader = CSVLoader()