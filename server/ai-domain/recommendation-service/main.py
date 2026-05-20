from fastapi import FastAPI
from app.api import recommendation

app = FastAPI(title="Recommendation Service")
app.include_router(recommendation.router, prefix="/api/v1", tags=["recommendation"])