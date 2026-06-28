"""
ML Tools V3

Automatic model training,
model comparison,
model selection,
and feature importance extraction.
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier


class MLTools:

    def train_models(self, X, y):

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # Candidate models
        models = {

            "Logistic Regression":
                LogisticRegression(
                    max_iter=1000
                ),

            "Decision Tree":
                DecisionTreeClassifier(),

            "Random Forest":
                RandomForestClassifier(),

            "XGBoost":
                XGBClassifier(
                    eval_metric='logloss'
                )
        }

        # Priority when scores tie
        model_priority = {

            "XGBoost": 4,
            "Random Forest": 3,
            "Decision Tree": 2,
            "Logistic Regression": 1
        }

        results = {}

        best_model = None
        best_name = None
        best_score = -1

        # Train all models
        for name, model in models.items():

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            results[name] = accuracy

            # Select best model
            if (
                accuracy > best_score
                or
                (
                    accuracy == best_score
                    and
                    model_priority[name]
                    >
                    model_priority.get(
                        best_name,
                        0
                    )
                )
            ):

                best_score = accuracy
                best_model = model
                best_name = name

        # Extract feature importance
        feature_importance = {}

        # Tree models
        if hasattr(
            best_model,
            "feature_importances_"
        ):

            feature_importance = dict(
                zip(
                    X.columns,
                    best_model.feature_importances_
                )
            )

        # Linear models
        elif hasattr(
            best_model,
            "coef_"
        ):

            feature_importance = dict(
                zip(
                    X.columns,
                    abs(
                        best_model.coef_[0]
                    )
                )
            )

        return {

            "best_model_name":
                best_name,

            "best_model":
                best_model,

            "best_accuracy":
                best_score,

            "all_results":
                results,

            "feature_importance":
                feature_importance
        }


# Singleton object
ml_tools = MLTools()