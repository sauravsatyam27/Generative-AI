import asyncio
import json

from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage


load_dotenv()


SERVERS = {

    # =========================
    # Math MCP Server
    # =========================
    "math": {
        "transport": "stdio",

        "command": r"E:\Gen AI\MCP\mcp-maths-server\.venv\Scripts\python.exe",

        "args": [
            r"E:\Gen AI\MCP\mcp-maths-server\src\mcp_maths_server\main.py"
        ],
    },

    # =========================
    # Expense MCP Server
    # =========================
    "expense": {
        "transport": "streamable_http",

        "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
    },

    # =========================
    # Manim MCP Server
    # =========================
    "manim-server": {
        "transport": "stdio",

        "command": "python",

        "args": [
            r"E:\Gen AI\MCP\manim-mcp-server\src\manim_server.py"
        ],

        "env": {
            "MANIM_EXECUTABLE":
                r"E:\Gen AI\MCP\manim-mcp-server\.venv\Scripts\manim.exe"
        }
    }
}


async def main():

    # ==========================================
    # Create MCP client
    # ==========================================

    client = MultiServerMCPClient(SERVERS)


    # ==========================================
    # Get all tools from all MCP servers
    # ==========================================

    tools = await client.get_tools()


    # ==========================================
    # Store tools by name
    # ==========================================

    named_tools = {}

    for tool in tools:
        named_tools[tool.name] = tool


    print("\nAvailable tools:")

    for tool_name in named_tools:
        print("-", tool_name)


    # ==========================================
    # Gemini
    # ==========================================

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )


    # ==========================================
    # Give MCP tools to Gemini
    # ==========================================

    llm_with_tools = llm.bind_tools(tools)


    # ==========================================
    # User prompt
    # ==========================================

    prompt = "Draw a triangle rotating in place using the manim tool."


    # ==========================================
    # First Gemini call
    # ==========================================

    response = await llm_with_tools.ainvoke(prompt)


    # ==========================================
    # Check whether Gemini called a tool
    # ==========================================

    if not getattr(response, "tool_calls", None):

        print("\nGemini Reply:")
        print(response.content)

        return


    # ==========================================
    # Execute Gemini's tool calls
    # ==========================================

    tool_messages = []


    for tc in response.tool_calls:

        selected_tool = tc["name"]

        selected_tool_args = tc.get("args") or {}

        selected_tool_id = tc["id"]


        print(f"\nCalling tool: {selected_tool}")

        print(f"Arguments: {selected_tool_args}")


        # Get MCP tool
        tool = named_tools[selected_tool]


        # Execute MCP tool
        result = await tool.ainvoke(
            selected_tool_args
        )


        print("Tool result:")
        print(result)


        # Create ToolMessage
        tool_messages.append(
            ToolMessage(
                tool_call_id=selected_tool_id,
                content=json.dumps(
                    result,
                    default=str
                )
            )
        )


    # ==========================================
    # Send tool result back to Gemini
    # ==========================================

    final_response = await llm_with_tools.ainvoke(
        [
            ("human", prompt),
            response,
            *tool_messages
        ]
    )


    # ==========================================
    # Final response
    # ==========================================

    print("\nFinal Gemini response:")

    print(final_response.content)


if __name__ == "__main__":
    asyncio.run(main())