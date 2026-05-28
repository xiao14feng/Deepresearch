import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

def search_web(query: str) -> tuple[list[dict], str]:
    """根据关键词搜索信息"""
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        result = client.search(query=query, max_results=3)

        results = []
        for item in result.get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "link": item.get("url", ""),
                    "snippet": item.get("content", ""),
                }
            )

        if len(results) == 0:
            return [], "Empty"

        return results, "Completed"

    except Exception as e:  
        return [], f"{str(e)}"

