"""
Prompt templates for
AI Data Scientist Copilot.
"""

PLANNER_PROMPT = """
You are an expert AI Data Scientist.

Create a detailed step-by-step
data science workflow.

Task:
{task}

Return only numbered steps.
"""


EDA_PROMPT = """
You are an expert data analyst.

Analyze the following dataset
statistics and provide insights.

Dataset Info:
{info}
"""


ML_PROMPT = """
You are an expert machine
learning engineer.

Suggest the best machine learning
algorithms for this dataset.

Dataset:
{info}
"""


REPORT_PROMPT = """
You are an expert business analyst.

Generate a professional report
from the following analysis.

Analysis:
{analysis}
"""