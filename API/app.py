from rich import print as rprint
from rich.panel import Panel
from fastapi import FastAPI, HTTPException
from ..core.graph import build_graph  
from langchain_core.messages import HumanMessage
from .schemas import ChatRequest , ChatResponse 

graph = build_graph()

app = FastAPI()

@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.message
    print('Received user input...')

    if not user_input.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    human_message = HumanMessage(content=user_input)
    
    try:
        config = {"configurable": {"thread_id": str(request.session_id)}}
        output = graph.invoke({"messages": [human_message], "question": user_input, "summary": request.summary}, config)
        response_text = str(output.get("generation",""))
        new_summary = str(output['summary'])

        rprint(Panel("AI: " + response_text))

        return ChatResponse(
            message=response_text,
            session_id=request.session_id,
            summary=new_summary,
            user_id=request.user_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))