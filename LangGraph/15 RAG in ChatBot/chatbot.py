from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.messages import (
    BaseMessage,
    HumanMessage
)

from langchain_core.tools import tool

from langgraph.graph import (
    StateGraph,
    START
)

from langgraph.graph.message import add_messages

from langgraph.prebuilt import (
    ToolNode,
    tools_condition
)

from typing import TypedDict, Annotated


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 2. CREATE GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# ============================================================
# 3. CREATE CALCULATOR TOOL
# ============================================================

@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str
) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.

    Supported operations:
    add, sub, mul, div, mod
    """

    try:

        # Addition
        if operation == "add":
            result = first_num + second_num

        # Subtraction
        elif operation == "sub":
            result = first_num - second_num

        # Multiplication
        elif operation == "mul":
            result = first_num * second_num

        # Division
        elif operation == "div":

            if second_num == 0:
                return {
                    "error": "Division by zero is not allowed"
                }

            result = first_num / second_num

        # Modulus
        elif operation == "mod":

            if second_num == 0:
                return {
                    "error": "Modulus by zero is not allowed"
                }

            result = first_num % second_num

        # Unsupported operation
        else:
            return {
                "error": f"Unsupported operation: {operation}"
            }

        # Return successful result
        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# ============================================================
# 4. ADD TOOLS
# ============================================================

tools = [
    calculator
]


# ============================================================
# 5. BIND TOOLS WITH GEMINI
# ============================================================

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 6. DEFINE STATE
# ============================================================

class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# ============================================================
# 7. CHAT NODE
# ============================================================

def chat_node(state: ChatState):

    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


# ============================================================
# 8. TOOL NODE
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# 9. CREATE GRAPH
# ============================================================

graph = StateGraph(ChatState)


# Add nodes

graph.add_node(
    "chat_node",
    chat_node
)

graph.add_node(
    "tools",
    tool_node
)


# ============================================================
# 10. CREATE EDGES
# ============================================================

# START → chat_node

graph.add_edge(
    START,
    "chat_node"
)


# chat_node → tools OR END
#
# tools_condition checks whether Gemini
# requested a tool call.

graph.add_conditional_edges(
    "chat_node",
    tools_condition
)


# tools → chat_node
#
# After the calculator executes,
# send the result back to Gemini.

graph.add_edge(
    "tools",
    "chat_node"
)


# ============================================================
# 11. COMPILE GRAPH
# ============================================================

chatbot = graph.compile()


# ============================================================
# 12. RUN GRAPH
# ============================================================

result = chatbot.invoke(
    {
        "messages": [
            HumanMessage(
                content=(
                    "Find the modulus of 132354 and 23 "
                    "and give the answer like a cricket commentator."
                )
            )
        ]
    }
)


# ============================================================
# 13. PRINT FINAL RESPONSE
# ============================================================

print(result["messages"][-1].content)