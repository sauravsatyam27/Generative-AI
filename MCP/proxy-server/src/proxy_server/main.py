from fastmcp.server import create_proxy

mcp = create_proxy(
    "https://fixed-lavender-bass.fastmcp.app/mcp",
    name="Satyam Server Proxy"
)

if __name__ == "__main__":
    mcp.run()