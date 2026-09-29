from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    message: str
    approval: str


def human_approval(state: State):

    print("AI: I want to perform an action.")

    approval = interrupt(
        "Do you approve this action? (yes/no)"
    )

    return {
        "approval": approval
    }


# Create graph
builder = StateGraph(State)

builder.add_node("human_approval", human_approval)

builder.add_edge(START, "human_approval")
builder.add_edge("human_approval", END)


# Checkpointer
checkpointer = InMemorySaver()

graph = builder.compile(
    checkpointer=checkpointer
)