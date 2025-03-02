from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    message: str
    summary: str
    session_id: int 
    user_id: Optional[int] = -1
    
class ChatResponse(BaseModel):
    message: str
    session_id: int
    summary: str
    user_id: Optional[int]