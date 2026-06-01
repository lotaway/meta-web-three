from pydantic import BaseModel
from typing import List, Optional


class RecommendRequest(BaseModel):
    user_id: str
    limit: int = 10
    scene: Optional[str] = "home"