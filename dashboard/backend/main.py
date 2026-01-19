from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.services.llm import llm_service
from app.services.spatial import spatial_service
from app.api.endpoints import router
import uvicorn
import logging
import sys

from sqlalchemy import create_engine, text
from app.core.config import settings

def check_database():
    """Checks database connectivity and table existence on startup."""
    print("\n[DB Check] Connecting to Database...")
    try:
        engine = create_engine(settings.DATABASE_URL)
        with engine.connect() as conn:
            # Check Connection
            conn.execute(text("SELECT 1"))
            print("✅ [DB Check] Connection Successful.")
            
            # Check Tables
            tables = ["neighborhoods", "h3_grid"]
            missing = []
            for t in tables:
                try:
                    res = conn.execute(text(f"SELECT count(*) FROM {t}")).scalar()
                    print(f"✅ [DB Check] Table '{t}' found: {res} rows.")
                    if res == 0:
                        print(f"⚠️ [DB Check] Table '{t}' is empty.")
                except Exception:
                    print(f"❌ [DB Check] Table '{t}' NOT found.")
                    missing.append(t)
            
            if missing:
                print("\n" + "="*50)
                print("⚠️  CRITICAL: MISSING TABLES")
                print(f"The following tables are missing: {', '.join(missing)}")
                print("Please run 'python upload_data.py' on the server to populate the database.")
                print("="*50 + "\n")

    except Exception as e:
        print(f"\n❌ [DB Check] Connection Failed: {e}")
        print("Please check your DATABASE_URL and ensure Postgres is running.\n")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Services
    check_database()
    
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


# Custom Log Filter to suppress /health checks and stack them
class StackedHealthCheckFilter(logging.Filter):
    _health_count = 0

    def filter(self, record):
        if hasattr(record, "args") and len(record.args) >= 3 and record.args[2] == "/health":
            StackedHealthCheckFilter._health_count += 1
            # Stack the log in place using carriage return
            sys.stderr.write(f"\rHealth Check Status: OK (x{StackedHealthCheckFilter._health_count})")
            sys.stderr.flush()
            return False
        return True

# Configure Logging Globally
logging.getLogger("uvicorn.access").addFilter(StackedHealthCheckFilter())

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
