import asyncio
import sys
from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():
    client = MultiServerMCPClient({
        "echo1": {                                           # ← 连接配置传给构造函数
            "command": sys.executable,
            "args": ["MCP_LangGraph.py"],
            "transport": "stdio",
        },
        "echo2":{
            "command": sys.executable,
            "args": ["MCP.py"],
            "transport": "stdio",
        }
    })
    
    tools1 = await client.get_tools(server_name="echo1")
    tools2 = await client.get_tools(server_name="echo2")      # ← 指定服务器名
    
    for tool in tools1:
        result = await tool.ainvoke({"message": "hello"})
        print(result[0]["text"])
    for tool in tools2:
        result = await tool.ainvoke({"message": "hello"})
        print(result[0]["text"])

asyncio.run(main())
