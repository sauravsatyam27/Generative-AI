from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (BaseMessage,HumanMessage)
from langchain_core.tools import tool
from langgraph.graph import (StateGraph,START)
from langgraph.graph.message import add_messages
from langgraph.prebuilt import (ToolNode,tools_condition)
from typing import TypedDict, Annotated
import asyncio
import os

from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)
expense_token = os.getenv("EXPENSE_MCP_TOKEN")

client = MultiServerMCPClient(
    {
        "math": {
            "transport": "stdio",
            "command": "uv",
            "args": [
                "--directory",
                r"E:\Gen AI\MCP\mcp-maths-server",
                "run",
                "fastmcp",
                "run",
                "src/mcp_maths_server/main.py",
            ],
        },
       "expense": {
            "transport": "streamable_http",
            "url": "https://fashionable-yellow-opossum.fastmcp.app/mcp",
            "headers": {
                "Authorization": f"Bearer {expense_token}"
            }
        }
    }
)


#tools = [calculator]
#llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


async def build_graph():

    tools = await client.get_tools()
    print(tools)

    async def chat_node(state: ChatState):

        messages = state["messages"]

        llm_with_tools = llm.bind_tools(tools)

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

    chatbot = await build_graph()

    result = await chatbot.ainvoke(
        {
             "messages": [
                 HumanMessage(
                    content=(
                        "Add an expense of 100 rupees for car ride today. "
                        "Then show me the total expenses for today."
                    )
                )
            ]
        }
    )

    print(result["messages"][-1].content)
if __name__ == "__main__":
    asyncio.run(main())