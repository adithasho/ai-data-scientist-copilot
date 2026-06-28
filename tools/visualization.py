"""
Visualization Tool
Creates plots automatically.
"""

import os
import matplotlib.pyplot as plt


class VisualizationTool:

    def __init__(self):

        os.makedirs(
            "outputs",
            exist_ok=True
        )

    def plot_numeric_columns(self, df):

        numeric_columns = (
            df.select_dtypes(
                include=["number"]
            ).columns
        )

        generated = []

        for column in numeric_columns:

            plt.figure(figsize=(6,4))

            df[column].hist()

            plt.title(column)

            filename = (
                f"outputs/{column}.png"
            )

            plt.savefig(filename)

            plt.close()

            generated.append(filename)

        return generated


visualizer = VisualizationTool()