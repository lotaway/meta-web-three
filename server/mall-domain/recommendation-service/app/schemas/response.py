from pydantic import BaseModel
from typing import List


class RecommendResponse(BaseModel):
    user_id: str
    items: List[str]
    score: List[float]