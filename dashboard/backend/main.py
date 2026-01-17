from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.services.llm import llm_service
from app.services.spatial import spatial_service
from app.api.endpoints import router
import uvicorn
import logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Services
    print("Initializing LLM Service...")
    llm_service.initialize_models()
    
    print("Initializing Spatial Service...")
    spatial_service.load_data()
    
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


# Custom Log Filter to suppress /health checks
class HealthCheckFilter(logging.Filter):
    def filter(self, record):
        return hasattr(record, "args") and len(record.args) >= 3 and record.args[2] != "/health"

# Configure Logging Globally
logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
