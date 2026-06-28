"""
Global configuration.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:

    def __init__(self):

        self.OPENROUTER_API_KEY = os.getenv(
            "OPENROUTER_API_KEY"
        )

        self.MODEL_NAME = os.getenv(
            "MODEL_NAME"
        )

        self.BASE_URL = os.getenv(
            "BASE_URL"
        )

        self.APP_NAME = os.getenv(
            "APP_NAME"
        )

        self.DEBUG = (
            os.getenv(
                "DEBUG",
                "False"
            ).lower() == "true"
        )


settings = Settings()