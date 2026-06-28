"""
Explainability Tool

SHAP-based explainability
for ML models.
"""

import shap
import numpy as np


class ExplainabilityTool:

    def explain(
        self,
        model,
        X
    ):

        try:

            # Tree models
            if (
                hasattr(
                    model,
                    "feature_importances_"
                )
            ):

                explainer = (
                    shap.TreeExplainer(
                        model,
                        feature_perturbation=
                        "tree_path_dependent"
                    )
                )

                shap_values = (
                    explainer.shap_values(
                        X
                    )
                )

            # Linear models
            elif (
                hasattr(
                    model,
                    "coef_"
                )
            ):

                explainer = (
                    shap.LinearExplainer(
                        model,
                        X
                    )
                )

                shap_values = (
                    explainer.shap_values(
                        X
                    )
                )

            else:

                return {}

            importance = {}

            values = np.abs(
                shap_values
            ).mean(axis=0)

            for feature, value in zip(
                X.columns,
                values
            ):

                importance[
                    feature
                ] = float(value)

            return importance

        except Exception as e:

            print(
                "SHAP Error:",
                str(e)
            )

            return {}


explainer = ExplainabilityTool()