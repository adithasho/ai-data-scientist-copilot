"""
Chat Agent
"""

from config.llm import ask_llm


class ChatAgent:

    def ask(

        self,

        question,

        data_info,

        eda_info,

        ml_info

    ):

        prompt = f"""
You are an expert AI Data Scientist.

You have analyzed a dataset.

DATASET:
{data_info}

EDA:
{eda_info}

MODEL:
{ml_info}

USER QUESTION:
{question}

Answer professionally.
Provide insights and recommendations.
"""

        return ask_llm(
            prompt
        )


chat_agent = (
    ChatAgent()
)