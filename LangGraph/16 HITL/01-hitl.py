from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from typing import TypedDict


# 1. State
class State(TypedDict):
    message: str
    approval: str


# 2. Node
def human_approval(state: State):

    print("AI: I want to perform an action.")

    # Graph yahan pause hoga
    approval = interrupt(
        "Do you approve this action? (yes/no)"
    )

    return {
        "approval": approval
    }


# 3. Graph
builder = StateGraph(State)

builder.add_node("human_approval", human_approval)

builder.add_edge(START, "human_approval")
builder.add_edge("human_approval", END)

graph = builder.compile()