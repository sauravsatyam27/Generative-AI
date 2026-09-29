from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver


# -------------------------
# State
# -------------------------

class State(TypedDict):
    message: str
    approval: str


# -------------------------
# HITL Node
# -------------------------

def human_approval(state: State):

    print("\nAI:", state["message"])

    approval = interrupt(
        "Do you approve this action? (yes/no)"
    )

    return {
        "approval": approval
    }


# -------------------------
# Build Graph
# -------------------------

builder = StateGraph(State)

builder.add_node(
    "human_approval",
    human_approval
)

builder.add_edge(
    START,
    "human_approval"
)

builder.add_edge(
    "human_approval",
    END
)


# -------------------------
# Checkpointer
# -------------------------

checkpointer = InMemorySaver()

graph = builder.compile(
    checkpointer=checkpointer
)


# -------------------------
# Thread ID
# -------------------------

config = {
    "configurable": {
        "thread_id": "user-1"
    }
}


# -------------------------
# First Run
# -------------------------

result = graph.invoke(
    {
        "message": "I want to delete your expense record.",
        "approval": ""
    },
    config
)

print("\nGraph paused.")


# -------------------------
# Human Input
# -------------------------

answer = input(
    "\nDo you approve? (yes/no): "
)


# -------------------------
# Resume Graph
# -------------------------

result = graph.invoke(
    Command(resume=answer),
    config
)


print("\nFinal Result:")
print(result)