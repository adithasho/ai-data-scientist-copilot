"""
AI Data Scientist Copilot
Professional Dashboard V2
"""

import streamlit as st
import pandas as pd

from agents.planner_agent import planner
from agents.data_agent import data_agent
from agents.eda_agent import eda_agent
from agents.ml_agent import ml_agent
from agents.report_agent import report_agent
from agents.router_agent import router

from tools.explainability import explainer
from memory.memory_manager import memory


# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="AI Data Scientist Copilot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Data Scientist Copilot")
st.caption(
    "Autonomous Multi-Agent Data Science System"
)

st.sidebar.title("AI Copilot")
st.sidebar.info(
    """
    Features:
    - LLM Planner
    - Data Agent
    - EDA Agent
    - Router Agent
    - AutoML
    - Explainability
    - Memory
    - Reporting
    """
)

# ------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file:

    try:

        df = pd.read_csv(uploaded_file)

        st.success("Dataset Loaded")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        if st.button(
            "🚀 Analyze Dataset"
        ):

            progress = st.progress(0)

            # ====================================
            # PLANNER
            # ====================================

            progress.progress(10)

            task = f"""
Analyze the uploaded dataset.

Rows: {len(df)}
Columns: {list(df.columns)}

Create a professional
data science workflow.
"""

            plan = planner.create_plan(task)

            # ====================================
            # DATA AGENT
            # ====================================

            progress.progress(25)

            data_info = (
                data_agent
                .analyze(df)
            )

            # ====================================
            # EDA
            # ====================================

            progress.progress(40)

            eda_info = (
                eda_agent
                .analyze(df)
            )

            # ====================================
            # ROUTER
            # ====================================

            progress.progress(50)

            route = (
                router.route(df)
            )

            # ====================================
            # ML
            # ====================================

            progress.progress(70)

            target = df.columns[-1]

            ml_results = (
                ml_agent.train(
                    df,
                    target
                )
            )

            # ====================================
            # SHAP
            # ====================================

            progress.progress(85)

            X = df.drop(
                target,
                axis=1
            )

            shap_results = (
                explainer.explain(
                    ml_results[
                        "best_model"
                    ],
                    X
                )
            )

            # ====================================
            # MEMORY
            # ====================================

            memory.save_run(
                uploaded_file.name,
                ml_results[
                    "best_model_name"
                ],
                ml_results[
                    "best_accuracy"
                ]
            )

            progress.progress(100)

            # ====================================
            # METRICS
            # ====================================

            st.header(
                "Dashboard"
            )

            c1, c2, c3, c4 = (
                st.columns(4)
            )

            c1.metric(
                "Rows",
                data_info["rows"]
            )

            c2.metric(
                "Columns",
                data_info["columns"]
            )

            c3.metric(
                "Task",
                route["task"]
            )

            c4.metric(
                "Best Model",
                ml_results[
                    "best_model_name"
                ]
            )

            # ====================================
            # TABS
            # ====================================

            tab1, tab2, tab3, tab4, tab5 = (
                st.tabs([
                    "AI Plan",
                    "EDA",
                    "Models",
                    "Explainability",
                    "Report"
                ])
            )

            # --------------------------------

            with tab1:

                st.markdown(plan)

            # --------------------------------

            with tab2:

                st.subheader(
                    "Dataset Summary"
                )

                st.json(
                    data_info
                )

                st.subheader(
                    "EDA"
                )

                st.json(
                    eda_info
                )

            # --------------------------------

            with tab3:

                st.subheader(
                    "Model Comparison"
                )

                results = (
                    pd.DataFrame(

                        ml_results[
                            "all_results"
                        ].items(),

                        columns=[
                            "Model",
                            "Accuracy"
                        ]
                    )
                )

                st.bar_chart(
                    results.set_index(
                        "Model"
                    )
                )

                st.success(

                    f"Best Model: "
                    f"{ml_results['best_model_name']}"
                )

                st.success(

                    f"Accuracy: "
                    f"{ml_results['best_accuracy']:.4f}"
                )

            # --------------------------------

            with tab4:

                st.subheader(
                    "Feature Importance"
                )

                if shap_results:

                    importance = (
                        pd.DataFrame(

                            shap_results.items(),

                            columns=[
                                "Feature",
                                "Importance"
                            ]
                        )
                    )

                else:

                    importance = (
                        pd.DataFrame(

                            ml_results[
                                "feature_importance"
                            ].items(),

                            columns=[
                                "Feature",
                                "Importance"
                            ]
                        )
                    )

                st.bar_chart(
                    importance.set_index(
                        "Feature"
                    )
                )

            # --------------------------------

            with tab5:

                report = (
                    report_agent
                    .generate(
                        data_info,
                        eda_info,
                        ml_results
                    )
                )

                st.code(report)

    except Exception as e:

        st.error(
            f"Error: {e}"
        )