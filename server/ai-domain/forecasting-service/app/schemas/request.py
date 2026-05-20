from pydantic import BaseModel
from typing import Optional


class ForecastRequest(BaseModel):
    product_id: str
    days: int = 7
    granularity: Optional[str] = "day"