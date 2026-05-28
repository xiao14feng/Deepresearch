PROMPT_PLAN = """
    请你将我提供给你的任务，拆解成三个任务，并且输出结构化的结果。请只输出这个json文本，不要输出额外的内容，我后续需要放到程序中执行，请你将输出格式化。请注意任务1、2、3分别是任务的名称，关键词就是用于搜索的时候使用的关键词一般是单个或者多个英文或中文字或者词。关键词可以有多个。比如说我要搜索Langgraph，title里面不要使用任务一、二、三，要说明具体的任务，关键词就是用来解决title的问题的。
    案例：[
        {
            "title":"LangGraph原理是什么",
            "query":"LangGraph 原理"
        },
        {
            "title":"LangGraph如何开发",
            "query":"LangGraph 开发 流程"
        },
        {
            "title":"LangGraph优势是哪里",
            "query":"LangGraph 对比 优势"
        }
    ]
"""