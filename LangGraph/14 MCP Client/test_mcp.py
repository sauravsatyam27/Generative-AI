import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient


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
        "expense":{
            "transport": "streamable_http",
            "url":"https://fashionable-yellow-opossum.fastmcp.app/mcp"
        }
    }
)


async def main():

    print("Connecting to MCP server...")

    tools = await client.get_tools()

    print("\nMCP connection successful!")

    print("\nAvailable tools:")

    for tool in tools:
        print(f"- {tool.name}")


if __name__ == "__main__":
    asyncio.run(main())