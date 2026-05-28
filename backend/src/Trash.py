def search_query(state: ResearchState):
    current_task_index = state["current_task_index"]
    todo_items = state.get("todo_items", [])
    task_results = state.get("task_results", [])
    rag_results = state.get("rag_results", [])
    title = state["title"]
    query = state["query"]

    # 使用Rag查询
    rag_texts = []
    rag_outputs = search(query)
    
    for key, output in enumerate(rag_outputs):
        text = output["text"]
        rag_texts.append(f"检索到的文档{key}, {text}\n")
    
    rag_result = {
        "title": title,
        "output": list(rag_texts)
    }

    # 使用Web查询
    web_texts = []
    web_outputs, error = search_web(query)    

    if error:
        web_texts = ["网络搜索失败"]
    else:
        for key, value in enumerate(web_outputs):
            titles = web_outputs[key]["title"]
            snippet = web_outputs[key]["snippet"]
            web_texts.append(f"网络搜索结果{key} - 标题：{titles} - 内容：{snippet}\n")

    web_result = {
        "title": title, 
        "outputs": list(web_texts)
    }

    return{
        "web_results": [web_result],
        "rag_results": [rag_result]
    }

def rag_task(state: ResearchState):
    """使用rag检索相关问题"""
    task_results = state.get("task_results", [])
    todo_items = state.get("todo_items", [])
    current_task_index = state["current_task_index"]
    rag_results = state.get("rag_results", [])

    texts = []
    title = todo_items[current_task_index]["title"]
    count = build_index()
    # print(f"索引数量: {count}")
    outputs = search(title)

    for key, output in enumerate(outputs):
        text = output["text"]
        texts.append(f"检索到的文档{key}, {text}\n")
        # print(f"检索到的文档{key}, {text}")

    index = current_task_index
    if texts:
        rag_results.append(
            {
                "title":title,
                "output":list(texts)
            }
        )
    else:
        index = index - 1
        rag_results.append(
            {
                "title":title,
                "output":"没有检索到结果"
            }
        )

    return {
        "rag_results": rag_results,
        "current_task_index": current_task_index+1
    }

def run_task(state: ResearchState):
    """执行网络搜索"""
    todo_items = state["todo_items"]
    current_task_index = state["current_task_index"] - 1
    task_results = state.get("task_results", "")
    query = todo_items[current_task_index]["query"]
    title = todo_items[current_task_index]["title"]

    content, error = search_web(query) 
    result = []
    finish = ""
    summary = ""

    for key, value in enumerate(content):
        titles = content[key]["title"]
        snippet = content[key]["snippet"]
        str = f"标题:{titles}\n内容:{snippet}"
        result.append(str)
    
    if len(result) == 0:
        finish = "未找到相关内容"
        task_results.append({
            "task_id": current_task_index,
            "title": title, 
            "query": query,
            "results": [],
            "summary": "没有搜索到相关信息",
            "status": finish,
            "error": error
        })
    else:
        sum = LLM.invoke(f"用200字总结一下这些内容:{result}")
        summary = sum.content
        finish = "已完成"
        task_results.append({
            "task_id": current_task_index,
            "title": title, 
            "query": query,
            "results": list(result),
            "summary": summary,
            "status": finish,
            "error": error
        })

    return {
        "task_results": task_results,
        "current_task_index": current_task_index 
    }

def should_continue(state: ResearchState):
    """判断是否继续执行任务"""
    current_task_index = state["current_task_index"] 
    todo_items = state["todo_items"]
    
    if current_task_index < len(todo_items):
        return "continue"
    else:
        return "end"
    
def should_web(state: ResearchState):
    """判断是否网络搜索"""
    rag_results = state.get("rag_results", [])
    current_task_index = state["current_task_index"]
    todo_items = state["todo_items"]
    
    output = rag_results[current_task_index - 1]["output"]
    if len(output) == 0:
        return "web"
    elif current_task_index < len(todo_items):
        return "rag"
    else:
        return "end"