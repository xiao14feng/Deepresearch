import json
import asyncio
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from src.graph import agent
from src.rag.indexing import build_index

def format_sse(event: str, data: dict):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=5, description="研究主题")

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(asyncio.to_thread(build_index))
    yield
    await task

app = FastAPI(lifespan=lifespan)

@app.get("/")
def index():
    return FileResponse(Path(__file__).parent.parent / "fronted" / "index.html")

@app.post("/research")
def research(request: ResearchRequest):
    try:
        result = agent.invoke({"topic": request.topic})
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_type": "AgentRunError",
                "message": f"研究失败:{str(e)}"
            }
        )

    return {
        "todo_items": result["todo_items"],
        "task_results": result["task_results"],
        "rag_results": result["rag_results"],
        "final_report": result["final_report"]
    }

@app.post("/research/stream")
def research_stream(request: ResearchRequest):
    def event_generator():
        yield format_sse("start", {"topic": request.topic})

        for chunk in agent.stream(
            {"topic": request.topic},
            stream_mode="updates"
        ):
            for node_name, node_output in chunk.items():
                yield format_sse(node_name, node_output)
        
        yield format_sse("done", {"success": True})
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )

@app.get("/healthz")
def healthz():
    return {"status": "ok"}