"""
Report Agent
"""


class ReportAgent:

    def generate(
        self,
        data_info,
        eda_info,
        ml_info
    ):

        report = f"""
=========================================
AI DATA SCIENTIST COPILOT REPORT
=========================================

DATASET INFORMATION
-------------------

Rows:
{data_info['rows']}

Columns:
{data_info['columns']}

Numerical Features:
{data_info['numerical_columns']}

Categorical Features:
{data_info['categorical_columns']}


EDA SUMMARY
-----------

Missing Values:
{eda_info['missing_values']}

Duplicates:
{eda_info['duplicates']}


MODEL PERFORMANCE
-----------------
Best Model:
{ml_info['best_model_name']}

Accuracy:
{ml_info['best_accuracy']:.4f}

All Models:
{ml_info['all_results']}

Feature Importance:
{ml_info['feature_importance']}

ALL MODEL RESULTS
-----------------

{ml_info['all_results']}


FEATURE IMPORTANCE
------------------

{ml_info['feature_importance']}


CONCLUSION
----------

The AI Data Scientist Copilot
successfully analyzed the dataset,
selected the optimal model,
and generated explainability insights.
"""

        return report


report_agent = ReportAgent()