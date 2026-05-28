import operator
from typing import TypedDict, Annotated


class ResearchState(TypedDict):
    topic: str                  # 主题
    max_attempts: int           # 最大尝试次数
    errors: Annotated[list[str], operator.add]                # 多个分支失败原因
    title: str
    query: str
    todo_items: list[dict]      # 任务列表
    current_task_index: int     # 当前任务

    task_results: Annotated[list[dict], operator.add]    # 网络搜索结果
    rag_results: Annotated[list[dict], operator.add]     # RAG搜索结果

    times: list[dict]            # 记录时间
    final_report: str           # 最终报告