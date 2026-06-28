"""
Preprocessing Agent
"""

import pandas as pd

from sklearn.preprocessing import LabelEncoder


class PreprocessingAgent:

    def preprocess(
        self,
        df,
        target
    ):

        df = df.copy()

        # Remove ID columns
        for column in list(
            df.columns
        ):

            if (

                "id"
                in
                column.lower()

                or

                df[column]
                .nunique()
                ==
                len(df)
            ):

                if (
                    column
                    != target
                ):

                    df = df.drop(
                        column,
                        axis=1
                    )

        # Handle missing values
        for column in df.columns:

            if (

                pd.api.types
                .is_object_dtype(
                    df[column]
                )

                or

                pd.api.types
                .is_string_dtype(
                    df[column]
                )

                or

                pd.api.types
                .is_categorical_dtype(
                    df[column]
                )
            ):

                df[column] = (
                    df[column]
                    .fillna(
                        "Unknown"
                    )
                )

            else:

                df[column] = (
                    df[column]
                    .fillna(
                        df[column]
                        .median()
                    )
                )

        # Encode categories
        encoder = (
            LabelEncoder()
        )

        for column in df.columns:

            if (
                column
                != target
            ):

                if (

                    pd.api.types
                    .is_object_dtype(
                        df[column]
                    )

                    or

                    pd.api.types
                    .is_string_dtype(
                        df[column]
                    )

                    or

                    pd.api.types
                    .is_categorical_dtype(
                        df[column]
                    )
                ):

                    df[column] = (
                        encoder
                        .fit_transform(
                            df[column]
                        )
                    )

        # Encode target
        if (

            pd.api.types
            .is_object_dtype(
                df[target]
            )

            or

            pd.api.types
            .is_string_dtype(
                df[target]
            )
        ):

            df[target] = (
                encoder
                .fit_transform(
                    df[target]
                )
            )

        return df


preprocessor = (
    PreprocessingAgent()
)