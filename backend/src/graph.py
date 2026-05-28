from langgraph.graph import END, START, StateGraph
from .state import ResearchState
# from .nodes import generate_report, plan_tasks, run_task,should_continue,rag_task, should_web, search_query, assign_task, search_mcp_tool
from .nodes import generate_report, search_mcp_tool, plan_tasks, assign_task
graph = StateGraph(ResearchState)
graph.add_node("generate", generate_report)
graph.add_node("plan", plan_tasks)
graph.add_node("search", search_mcp_tool)


graph.add_edge(START, "plan")
graph.add_conditional_edges(
    "plan",
    assign_task,
    {
        "end": END,
        "search": "search",
        "generate": "generate"
    }
)
graph.add_edge("search", "generate")
graph.add_edge("generate", END)

agent = graph.compile()