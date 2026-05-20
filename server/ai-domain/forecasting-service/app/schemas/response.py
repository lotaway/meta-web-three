from pydantic import BaseModel
from typing import List


class ForecastResponse(BaseModel):
    product_id: str
    forecast_days: int
    values: List[float]