
from fastapi import FastAPI
from pydantic import BaseModel
from src.models import AgentResponse
from src.agent.agent import answer

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

# Create Post Endpoint
@app.post("/api/chat")
async def sent_chat(req: ChatRequest) -> AgentResponse:
    return await answer(req.question)