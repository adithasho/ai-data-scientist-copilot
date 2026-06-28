"""
LLM Connection
"""

from openai import OpenAI
from config.settings import settings


client = OpenAI(
    api_key=settings.OPENROUTER_API_KEY,
    base_url=settings.BASE_URL
)


def ask_llm(prompt):

    response = client.chat.completions.create(
        model=settings.MODEL_NAME,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content