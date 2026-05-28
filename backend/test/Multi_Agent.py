from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from .state import ResearchState
from .config import LLM
from .tools import search_web
from .rag import search, build_index
import json


def clean_json_text(text: str):
    text = text.strip()
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
        text = text[start: end + 1]
    return text


# ============= 节点 =============


def supervisor_node(state: ResearchState):
    """管理者：制定计划 / 推进任务索引"""
    todo_items = state.get("todo_items", [])
    current_index = state.get("current_task_index", 0)
    rag_results = state.get("rag_results", [])
    topic = state["topic"]

    # 第一次调用 → 制定计划
    if not todo_items:
        
        prompt = f"""请将研究主题「{topic}」拆解成3个具体的子问题。
要求: 每个子问题可以独立研究，符合MECE原则，使用中文
格式: JSON数组 [{{"title": "子问题标题", "query": "搜索关键词"}}]
只返回JSON，不要其他内容"""
        result = LLM.invoke(prompt)
        tasks = json.loads(clean_json_text(result.content))
        print(f"[管理者] 计划已制定: {[t['title'] for t in tasks]}")
        return {
            "todo_items": tasks,
            "current_task_index": 0,
            "task_results": [],
            "rag_results": [],
            "final_report": ""
        }

    # 有研究结果回来了 → 推进索引
    if current_index < len(rag_results):
        print(f"[管理者] 任务 {current_index} 完成，推进到 {current_index + 1}")
        return {"current_task_index": current_index + 1}

    print(f"[管理者] 等待研究结果 (index={current_index})")
    return {}


def supervisor_router(state: ResearchState):
    """路由：决定 supervisor 下一步"""
    todo_items = state.get("todo_items", [])
    current_index = state.get("current_task_index", 0)
    rag_results = state.get("rag_results", [])

    # 还没计划 → 继续等
    if not todo_items:
        return "supervisor"

    # 全部完成 → 写报告
    if current_index >= len(todo_items):
        print(f"[路由] 所有任务完成，进入生成")
        return "generate"

    # # 当前任务还没研究 → 派给研究员
    # if current_index >= len(rag_results):
    #     task = todo_items[current_index]
    #     print(f"[路由] 派发任务: {task['title']}")
    #     return Send("research", {
    #         "topic": state["topic"],
    #         "title": task["title"],
    #         "query": task["query"],
    #         "todo_items": todo_items,
    #         "task_results": state.get("task_results", []),
    #         "rag_results": rag_results,
    #         "current_task_index": current_index,
    #         "final_report": state.get("final_report", "")
    #     })

    if current_index == 0 and len(rag_results) == 0:
        sends = []
        for i, task in enumerate(todo_items):
            sends.append(Send("research",{
                "title": task["title"],
                "query": task["query"],
                "task_index": i
            }))
        return sends

    # 有结果但索引还没推进 → 回 supervisor 推进
    return "supervisor"


def research_node(state: ResearchState):
    """研究员：执行 RAG + Web 搜索"""
    query = state.get("query", "")
    title = state.get("title", "")
    current_index = state.get("current_task_index", 0)
    rag_results = list(state.get("rag_results", []))

    print(f"[研究员] 开始查询: {title}")

    docs = search(query, k=3)

    texts = []
    if docs:
        for d in docs:
            texts.append(f"[知识库-{d['source']}] {d['text'][:300]}")
        source = "rag"
    else:
        # RAG 没有 → Web 搜索
        web_results, error = search_web(query)
        for w in web_results:
            texts.append(f"[网络] {w['title']}: {w['snippet'][:200]}")
        source = "web"

    rags_results = {
        "title": title,
        "output": texts,
        "source": source
    }

    print(f"[研究员] 完成: {title} ({source}, 共{len(texts)}条)")
    print(rags_results)
    return {"rag_results": [rags_results]}

def generate_node(state: ResearchState):
    """报告生成：汇总所有研究结果"""
    topic = state["topic"]
    todo_items = state.get("todo_items", [])
    rag_results = state.get("rag_results", [])

    print(f"[报告生成] 开始汇总，共 {len(todo_items)} 个子任务")

    parts = []
    for i, task in enumerate(todo_items):
        title = task.get("title", "")
        if i < len(rag_results):
            output = rag_results[i].get("output", [])
            output_text = "\n".join(output[:3]) if output else "无结果"
        else:
            output_text = "无结果"
        parts.append(f"## {title}\n{output_text}")

    all_text = "\n\n".join(parts)
    prompt = f"""主题: {topic}
以下是研究资料:
{all_text}

请基于以上资料，写一份不超过100字的结构清晰的研究报告。
要求: 使用中文，分段阐述，每个主题下要有具体内容。"""
    result = LLM.invoke(prompt)

    print(f"[报告生成] 完成")
    return {"final_report": result.content}


# ============= 构建 Graph =============
total = build_index()
graph = StateGraph(ResearchState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("research", research_node)
graph.add_node("generate", generate_node)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "supervisor": "supervisor",
        "research": "research",
        "generate": "generate"
    }
)
graph.add_edge("research", "supervisor")
graph.add_edge("generate", END)

agent = graph.compile()

if __name__ == "__main__":
    result = agent.invoke({"topic": "LangGraph的原理和优势"})
    print("\n=== 最终报告 ===")
    print(result["final_report"])
