from fastapi import APIRouter
from app.schemas.request import ForecastRequest
from app.schemas.response import ForecastResponse

router = APIRouter()


@router.post("/forecast", response_model=ForecastResponse)
async def get_forecast(req: ForecastRequest):
    return ForecastResponse(product_id=req.product_id, forecast_days=req.days, values=[])