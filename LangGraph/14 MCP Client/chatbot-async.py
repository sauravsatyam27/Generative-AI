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

import asyncio


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# ============================================================
# CALCULATOR TOOL
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

        # Successful result
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


tools = [calculator]


llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


def build_graph():


    async def chat_node(state: ChatState):

        messages = state["messages"]

        response = await llm_with_tools.ainvoke(messages)

        return {
            "messages": [response]
        }

    tool_node = ToolNode(tools)


    graph = StateGraph(ChatState)


    graph.add_node(
        "chat_node",
        chat_node
    )

    graph.add_node(
        "tools",
        tool_node
    )


    graph.add_edge(
        START,
        "chat_node"
    )

    graph.add_conditional_edges(
        "chat_node",
        tools_condition
    )

    graph.add_edge(
        "tools",
        "chat_node"
    )


    chatbot = graph.compile()

    return chatbot


async def main():

    chatbot = build_graph()

    result = await chatbot.ainvoke(
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

    print(result["messages"][-1].content)
if __name__ == "__main__":
    asyncio.run(main())