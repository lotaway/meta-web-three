from fastapi import FastAPI
from app.api import forecast

app = FastAPI(title="Forecasting Service")
app.include_router(forecast.router, prefix="/api/v1", tags=["forecast"])