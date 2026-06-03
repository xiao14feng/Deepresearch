import sys
import time
import json
import asyncio
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from langgraph.types import Send
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from src.state import ResearchState
from src.prompts import PROMPT_PLAN
from src.config import LLM
from src.tools import search_web
from src.rag.indexing import simple_search, build_index

_tools_cache = None
class JsonError(ValueError):
    pass
class RequiredError(ValueError):
    pass

def _load_tools():
    global _tools_cache
    if _tools_cache is None:
        async def load():
            client = MultiServerMCPClient({
                "search":{
                    "command": sys.executable,
                    "args": ["src/mcp_server.py"],
                    "transport": "stdio",
                }
            })
            return await client.get_tools(server_name="search")
        _tools_cache = asyncio.run(load())
    return _tools_cache
        
def search_mcp_tool(state: ResearchState):
    # times = state["times"]
    title = state["title"]
    query = state["query"]
    rag_results = []
    web_results = []

    tools = _load_tools()
    for tool in tools:
        try:
            if tool.name == "search_rag":
                rag_result = asyncio.run(tool.ainvoke({"title": title, "query": query}))
                rag_results.append(json.loads(rag_result[0]["text"]))

            if tool.name == "search_web1":
                task_result = asyncio.run(tool.ainvoke({"title": title, "query": query}))
                web_results.append(json.loads(task_result[0]["text"]))
        except Exception as e:
            traceback.print_exc()
            return{
                "rag_results": rag_results or [],
                "task_results": web_results or [],
                "errors": [f"工具{tool.name} 调用失败:{e}"]
            }
        
    return {
        "rag_results": rag_results,
        "task_results": web_results,
    }

def clean_json_text(text: str):
    text = text.strip()

    # 清除markdown格式的json文本
    if text.startswith("```json"):
        text = text[len("```json"):]
    elif text.startswith("```"):
        text = text[len("```"):]
    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1:
        text = text[start: end+1]

    return text

def plan_tasks(state: ResearchState):
    """将任务拆解成制定数量的子任务"""
    topic = state["topic"]
    max_test = state.get("max_attempt", 3)
    times = state.get("times", [])
    error = ""
    t0 = time.time()
    a0 = "拆解任务"
    times.append({
        "Action": a0,
        "Time": t0,
        "Total": 0
    })

    for i in range(max_test):
        try:
            prompt = f"{PROMPT_PLAN}，然后查询主题是{topic}"
            content = LLM.invoke(error + prompt).content
            text = clean_json_text(content)
            task = json.loads(text)
            if not isinstance(task, list):
                raise JsonError

            required = ["title", "query"]
            missing = [i for i, t in enumerate(task)
                if not isinstance(t, dict) or not all(k in t for k in required)]
            if missing:
                raise RequiredError
           
            break

        except JsonError:
            error = f"生成的json内容有问题，请重新生成只包含json文档的数据，这是你上一轮生成失败的内容{task}"
        
        except RequiredError:
            error = f"生成的json内容缺字段，需要的字段有{required}，你上一轮生成的错误内容是{task}"
        
    if not isinstance(task, list):    
        task = []
    
    t1 = time.time() - t0
    a1 = "拆解完毕"
    times.append({
        "Action": a1,
        "Time": time.time(),
        "Total": t1
    })
    

    return{
        "todo_items": task,
        "current_task_index":0,
        "task_results": [],
        "rag_results": [],
        "times": times
    }

def generate_report(state: ResearchState):
    """生成最后报告"""
    print("生成报告")
    times = state["times"]
    task_results = state.get("task_results", [])
    rag_results = state.get("rag_results", [])
    todo_items = state["todo_items"]
    topic = state["topic"]
    
    results = []
    titles = []

    for i in range(len(todo_items)):
        title = todo_items[i]["title"]
        rags = rag_results[i]["output"]
        webs = task_results[i]["output"]

        know = f"主题:{title},\n知识库数据:{list(rags)}, \n互联网搜索数据:{list(webs)}\n"
        sum_know = LLM.invoke(f"请用100个字给我总结{know}")

        results.append({
            "title": title,
            "content": sum_know.content
        })
    
    content = LLM.invoke(f"给我用50个字总结一下{results}").content
    a3 = "生成完成"
    t3 = time.time() - times[0]["Time"]
    times.append({
        "Action": a3,
        "Time": time.time(),
        "Total": t3
    })
    for t in times:
        print(f"{t['Action']}:{t['Total']}")

    return {
        "times": times,
        "final_report": content
    }

def assign_task(state: ResearchState):
    todo_items = state.get("todo_items", [])
    if todo_items == []:
        return "end"
    current_task_index = state["current_task_index"]
    print(f"[DEBUG search] todo_items={len(todo_items)}, index={current_task_index}, received_title={state.get('title')}") 

    sum1 = len(todo_items)
    if current_task_index >= sum1:
        return "generate"
    else: 
        Sends = []
        for i in range(sum1):
            Sends.append(
                Send("search",
                {
                    "title": todo_items[i]["title"],
                    "query": todo_items[i]["query"],
                    "current_task_index": i
                }
            ))
        return Sends
