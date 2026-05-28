from mcp.server import FastMCP

server = FastMCP("回声工具")

@server.tool()
def echo(message: str) -> str:
    return f"这是第二个MCP：{message}"

if __name__ == "__main__":
    server.run(transport="stdio")