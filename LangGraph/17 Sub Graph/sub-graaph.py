from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# =========================================================
# 1. SUBGRAPH STATE
# =========================================================

class SubState(TypedDict):
    message: str


# =========================================================
# 2. SUBGRAPH NODES
# =========================================================

def node_a(state: SubState):
    print("➡️ Subgraph: Node A")

    return {
        "message": state["message"] + " -> A"
    }


def node_b(state: SubState):
    print("➡️ Subgraph: Node B")

    return {
        "message": state["message"] + " -> B"
    }


# =========================================================
# 3. CREATE SUBGRAPH
# =========================================================

sub_builder = StateGraph(SubState)


# Add nodes
sub_builder.add_node("node_a", node_a)
sub_builder.add_node("node_b", node_b)


# Add edges
sub_builder.add_edge(START, "node_a")
sub_builder.add_edge("node_a", "node_b")
sub_builder.add_edge("node_b", END)


# Compile subgraph
subgraph = sub_builder.compile()


# =========================================================
# 4. MAIN GRAPH STATE
# =========================================================

class MainState(TypedDict):
    message: str


# =========================================================
# 5. MAIN GRAPH NODE
# =========================================================

def main_node(state: MainState):

    print("➡️ Main Graph: Main Node")

    return {
        "message": state["message"] + " -> Main"
    }


# =========================================================
# 6. CREATE MAIN GRAPH
# =========================================================

main_builder = StateGraph(MainState)


# Normal node
main_builder.add_node(
    "main_node",
    main_node
)


# Add SUBGRAPH as a node
main_builder.add_node(
    "subgraph",
    subgraph
)


# =========================================================
# 7. MAIN GRAPH EDGES
# =========================================================

main_builder.add_edge(
    START,
    "main_node"
)

main_builder.add_edge(
    "main_node",
    "subgraph"
)

main_builder.add_edge(
    "subgraph",
    END
)


# =========================================================
# 8. COMPILE MAIN GRAPH
# =========================================================

main_graph = main_builder.compile()


# =========================================================
# 9. RUN MAIN GRAPH
# =========================================================

result = main_graph.invoke({
    "message": "START"
})


# =========================================================
# 10. FINAL RESULT
# =========================================================

print("\n====================")
print("FINAL RESULT")
print("====================")

print(result)