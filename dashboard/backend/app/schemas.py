from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class AgentResponse(BaseModel):
    """Generic response structure for agents."""
    reasoning: str = Field(..., description="Step-by-step reasoning for the decision.")
    action: str = Field(..., description="Action taken (e.g., 'QUERY', 'PLAN', 'VIZ', 'ANSWER').")
    data: Any = Field(None, description="Structured data payload (GeoJSON, SQL result, etc.).")

class PlanStep(BaseModel):
    step_number: int
    description: str

class Plan(BaseModel):
    """Structured plan proposal."""
    reasoning: str = Field(..., description="Analysis of data availability and approach.")
    steps: List[str] = Field(..., description="List of execution steps.")
    confirmation_question: str = Field(..., description="Question to ask user for approval.")
    
class VizConfig(BaseModel):
    """Configuration for Deck.gl layers."""
    instruction: str = Field(..., description="Explanation of what is being shown.")
    layers: List[str] = Field(..., description="List of layer tags to enable (e.g., 'NEIGHBORHOODS').")
    view_state: Optional[Dict[str, Any]] = Field(None, description="Optional camera update (lat, lon, zoom).")
