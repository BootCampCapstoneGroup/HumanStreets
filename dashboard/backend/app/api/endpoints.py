from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from app.agents.router import RouterAgent
# llm_service import not strictly needed here if RouterAgent handles it, but maybe for cleanups
from app.services.llm import llm_service
from app.services.spatial import spatial_service

router = APIRouter()
router_agent = RouterAgent()

class ChatRequest(BaseModel):
    message: str
    latitude: float = Field(None, description="User latitude")
    longitude: float = Field(None, description="User longitude")
    model_provider: str = Field("local", description="Model provider: 'local', 'gemini', 'openrouter_free' or 'deepseek_free'")
    history: list[dict] = Field(default=[], description="Chat history for context")

@router.post("/chat")
async def chat(request: ChatRequest):
    user_msg = request.message
    loc_context = spatial_service.get_location_context(request.latitude, request.longitude)
    
    # We delegate everything to the RouterAgent
    # The RouterAgent will decide to use SQL, Viz, or Chart agents.
    
    try:
        return StreamingResponse(
            router_agent.route_and_execute(
                user_msg, 
                history=request.history,
                context={"location": loc_context}, 
                provider=request.model_provider
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
def health_check():
    # Basic check for now
    return {"status": "ok"}

@router.get("/api/layers/neighborhoods")
async def get_neighborhoods():
    try:
        return spatial_service.get_neighborhoods_geojson()
    except Exception as e:
         print(f"Error loading neighborhoods: {e}")
         raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/layers/h3")
async def get_h3_layer():
    try:
        return spatial_service.get_h3_records()
    except Exception as e:
        print(f"Error loading H3 data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/layers/query_result")
async def get_query_result_layer():
    """Returns the result of the last SQL query executed by the agent."""
    data = spatial_service.get_query_result()
    if not data:
        return {"type": "FeatureCollection", "features": []}
    return data
