"""
LangGraph Workflow
"""

from typing import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph import END

from agents.planner_agent import planner
from agents.data_agent import data_agent
from agents.eda_agent import eda_agent
from agents.ml_agent import ml_agent
from agents.report_agent import report_agent

from tools.csv_loader import csv_loader


class AgentState(TypedDict):

    task: str
    dataset: str

    dataframe: object

    plan: str

    data_info: dict
    eda_info: dict
    ml_info: dict

    report: str


def planner_node(state):

    print("\nPLANNER")

    state["plan"] = planner.create_plan(
        state["task"]
    )

    return state


def data_node(state):

    print("\nDATA AGENT")

    df = csv_loader.load(
        state["dataset"]
    )

    state["dataframe"] = df

    state["data_info"] = (
        data_agent.analyze(df)
    )

    return state


def eda_node(state):

    print("\nEDA AGENT")

    state["eda_info"] = (
        eda_agent.analyze(
            state["dataframe"]
        )
    )

    return state


def ml_node(state):

    print("\nML AGENT")

    target = (
        state["dataframe"]
        .columns[-1]
    )

    state["ml_info"] = (
        ml_agent.train(
            state["dataframe"],
            target
        )
    )

    return state


def report_node(state):

    print("\nREPORT AGENT")

    state["report"] = (
        report_agent.generate(
            state["data_info"],
            state["eda_info"],
            state["ml_info"]
        )
    )

    return state


workflow = StateGraph(
    AgentState
)

workflow.add_node(
    "planner",
    planner_node
)

workflow.add_node(
    "data",
    data_node
)

workflow.add_node(
    "eda",
    eda_node
)

workflow.add_node(
    "ml",
    ml_node
)

workflow.add_node(
    "report",
    report_node
)

workflow.set_entry_point(
    "planner"
)

workflow.add_edge(
    "planner",
    "data"
)

workflow.add_edge(
    "data",
    "eda"
)

workflow.add_edge(
    "eda",
    "ml"
)

workflow.add_edge(
    "ml",
    "report"
)

workflow.add_edge(
    "report",
    END
)

graph = workflow.compile()