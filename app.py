"""
AI Data Scientist Copilot
Professional Dashboard V3
"""

import streamlit as st
import pandas as pd

from agents.planner_agent import planner
from agents.data_agent import data_agent
from agents.eda_agent import eda_agent
from agents.ml_agent import ml_agent
from agents.report_agent import report_agent
from agents.router_agent import router
from agents.autonomous_agent import autonomous_agent

from tools.explainability import explainer
from memory.memory_manager import memory


# ==================================================
# SESSION STATE
# ==================================================

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "results" not in st.session_state:
    st.session_state.results = {}

if "df" not in st.session_state:
    st.session_state.df = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="AI Data Scientist Copilot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Data Scientist Copilot")
st.caption(
    "Autonomous Multi-Agent Data Science System"
)


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("AI Copilot")

st.sidebar.info(
    """
### Features

✓ LLM Planner

✓ Data Agent

✓ EDA Agent

✓ Router Agent

✓ AutoML

✓ Explainability

✓ Agent Memory

✓ Autonomous AI Agent

✓ Report Generation
"""
)


# ==================================================
# FILE UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file:

    try:

        df = pd.read_csv(uploaded_file)

        st.session_state.df = df

        st.success(
            "Dataset Loaded"
        )

        st.subheader(
            "Dataset Preview"
        )

        st.dataframe(
            df.head()
        )

        # ==========================================
        # ANALYZE BUTTON
        # ==========================================

        if st.button(
            "🚀 Analyze Dataset"
        ):

            st.session_state.analysis_done = True

            progress = st.progress(0)

            # ======================================
            # PLANNER
            # ======================================

            progress.progress(10)

            task = f"""
Analyze the uploaded dataset.

Rows: {len(df)}

Columns:
{list(df.columns)}

Create a professional
data science workflow.
"""

            plan = planner.create_plan(
                task
            )

            # ======================================
            # DATA AGENT
            # ======================================

            progress.progress(25)

            data_info = (
                data_agent
                .analyze(df)
            )

            # ======================================
            # EDA
            # ======================================

            progress.progress(40)

            eda_info = (
                eda_agent
                .analyze(df)
            )

            # ======================================
            # ROUTER
            # ======================================

            progress.progress(50)

            route = (
                router.route(df)
            )

            # ======================================
            # ML
            # ======================================

            progress.progress(70)

            target = (
                df.columns[-1]
            )

            ml_results = (
                ml_agent.train(
                    df,
                    target
                )
            )

            # ======================================
            # SHAP
            # ======================================

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

            # ======================================
            # MEMORY
            # ======================================

            memory.save_run(

                uploaded_file.name,

                ml_results[
                    "best_model_name"
                ],

                ml_results[
                    "best_accuracy"
                ]
            )

            # ======================================
            # SAVE RESULTS
            # ======================================

            st.session_state.results = {

                "plan":
                    plan,

                "data_info":
                    data_info,

                "eda_info":
                    eda_info,

                "route":
                    route,

                "ml_results":
                    ml_results,

                "shap_results":
                    shap_results
            }

            progress.progress(100)

        # ==========================================
        # DISPLAY RESULTS
        # ==========================================

        if st.session_state.analysis_done:

            plan = (
                st.session_state
                .results["plan"]
            )

            data_info = (
                st.session_state
                .results["data_info"]
            )

            eda_info = (
                st.session_state
                .results["eda_info"]
            )

            route = (
                st.session_state
                .results["route"]
            )

            ml_results = (
                st.session_state
                .results["ml_results"]
            )

            shap_results = (
                st.session_state
                .results["shap_results"]
            )

            # ======================================
            # DASHBOARD
            # ======================================

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

            # ======================================
            # TABS
            # ======================================

            (
                tab1,
                tab2,
                tab3,
                tab4,
                tab5,
                tab6
            ) = st.tabs(

                [
                    "🧠 AI Plan",
                    "📊 EDA",
                    "🤖 Models",
                    "🔍 Explainability",
                    "💬 AI Agent",
                    "📄 Report"
                ]
            )

            # ======================================
            # PLAN
            # ======================================

            with tab1:

                st.markdown(plan)

            # ======================================
            # EDA
            # ======================================

            with tab2:

                st.json(data_info)

                st.json(eda_info)

            # ======================================
            # MODELS
            # ======================================

            with tab3:

                results = pd.DataFrame(

                    ml_results[
                        "all_results"
                    ].items(),

                    columns=[
                        "Model",
                        "Accuracy"
                    ]
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

            # ======================================
            # EXPLAINABILITY
            # ======================================

            with tab4:

                if shap_results:

                    importance = pd.DataFrame(

                        shap_results.items(),

                        columns=[
                            "Feature",
                            "Importance"
                        ]
                    )

                else:

                    importance = pd.DataFrame(

                        ml_results[
                            "feature_importance"
                        ].items(),

                        columns=[
                            "Feature",
                            "Importance"
                        ]
                    )

                st.bar_chart(
                    importance.set_index(
                        "Feature"
                    )
                )

            # ======================================
            # AUTONOMOUS AI AGENT
            # ======================================

            with tab5:

                st.subheader(
                    "🤖 Autonomous AI Data Scientist"
                )

                st.info(
                    "Ask questions about your dataset."
                )

                examples = [

                    "How many customers churned?",

                    "Show churned customers",

                    "Describe the dataset",

                    "Show missing values",

                    "Show correlations",

                    "Why did XGBoost perform best?",

                    "How can accuracy be improved?"
                ]

                for ex in examples:

                    st.write(
                        f"• {ex}"
                    )

                # CHAT HISTORY

                for chat in (
                    st.session_state
                    .chat_history
                ):

                    with st.chat_message(
                        "user"
                    ):

                        st.write(
                            chat["question"]
                        )

                    with st.chat_message(
                        "assistant"
                    ):

                        st.write(
                            chat["answer"]
                        )

                question = st.chat_input(
                    "Ask about your dataset..."
                )

                if question:

                    with st.chat_message(
                        "user"
                    ):

                        st.write(
                            question
                        )

                    with st.spinner(
                        "Thinking..."
                    ):

                        response = (
                            autonomous_agent
                            .run(

                                question,

                                st.session_state.df
                            )
                        )

                    with st.chat_message(
                        "assistant"
                    ):

                        st.write(
                            response[
                                "answer"
                            ]
                        )

                        with st.expander(
                            "Agent Reasoning"
                        ):

                            st.json(
                                response[
                                    "decision"
                                ]
                            )

                            st.write(
                                response[
                                    "result"
                                ]
                            )

                    st.session_state.chat_history.append(

                        {

                            "question":
                                question,

                            "answer":
                                response[
                                    "answer"
                                ]
                        }
                    )

            # ======================================
            # REPORT
            # ======================================

            with tab6:

                report = (
                    report_agent
                    .generate(

                        data_info,

                        eda_info,

                        ml_results
                    )
                )

                st.code(
                    report,
                    language="text"
                )

            st.sidebar.success(
                "Analysis Completed"
            )

    except Exception as e:

        st.error(
            str(e)
        )