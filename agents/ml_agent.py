"""
ML Agent

Handles machine learning
training and model
selection.
"""

from tools.ml_tools import ml_tools
from agents.preprocessing_agent import preprocessor


class MLAgent:

    def train(
        self,
        df,
        target
    ):

        # Preprocess dataset
        df = preprocessor.preprocess(
            df,
            target
        )

        # Separate features and target
        X = df.drop(
            target,
            axis=1
        )

        y = df[target]

        # Train models
        results = (
            ml_tools.train_models(
                X,
                y
            )
        )

        return results


ml_agent = MLAgent()