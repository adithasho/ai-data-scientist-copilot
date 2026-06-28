"""
Memory Manager

Stores previous analyses
performed by the AI
Data Scientist Copilot.
"""

import json
import os


class MemoryManager:

    def __init__(self):

        self.memory_file = (
            "memory/history.json"
        )

        if not os.path.exists(
            self.memory_file
        ):

            with open(
                self.memory_file,
                "w"
            ) as f:

                json.dump(
                    [],
                    f
                )

    def save_run(
        self,
        dataset,
        model,
        accuracy
    ):

        history = self.load_history()

        history.append({

            "dataset":
                dataset,

            "best_model":
                model,

            "accuracy":
                float(
                    accuracy
                )
        })

        with open(
            self.memory_file,
            "w"
        ) as f:

            json.dump(
                history,
                f,
                indent=4
            )

    def load_history(self):

        with open(
            self.memory_file,
            "r"
        ) as f:

            return json.load(f)

    def get_best_run(self):

        history = self.load_history()

        if not history:

            return None

        return max(

            history,

            key=lambda x:
            x["accuracy"]
        )


memory = MemoryManager()