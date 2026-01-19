import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:12345@localhost:5432/capstone")
    
    # Dynamic Base Dir (points to HumanStreets root)
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    H3_DATA_PATH = os.path.join(BASE_DIR, "data", "riyadh_h3_r9.parquet")
    NEIGHBORHOODS_PATH = os.path.join(BASE_DIR, "data", "riyadh_neighborhoods.geojson")
    
    MODEL_ID = "LiquidAI/LFM2-1.2B"
    ADAPTER_PATH = os.path.join(BASE_DIR, "dashboard", "backend", "checkpoint-226")

settings = Config()
