from fastapi import APIRouter
from app.schemas.request import RecommendRequest
from app.schemas.response import RecommendResponse

router = APIRouter()


@router.post("/recommend", response_model=RecommendResponse)
async def get_recommendations(req: RecommendRequest):
    return RecommendResponse(user_id=req.user_id, items=[], score=[])