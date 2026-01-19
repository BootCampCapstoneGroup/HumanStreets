import os
import sys
# Ensure we can import app modules
sys.path.append(os.getcwd())

from sqlalchemy import create_engine, text
from app.core.config import settings

print(f"DEBUG: Using DATABASE_URL from settings: {settings.DATABASE_URL}")

try:
    engine = create_engine(settings.DATABASE_URL)
    with engine.connect() as conn:
        print("--- Executing Direct SQL ---")
        sql = "SELECT COUNT(*) FROM neighborhoods WHERE avg_walkability > 50;"
        print(f"Query: {sql}")
        result = conn.execute(text(sql)).scalar()
        print(f"Result: {result}")
        
        sql2 = "SELECT count(*) FROM neighborhoods WHERE avg_walkability > 80;"
        print(f"Query: {sql2}")
        result2 = conn.execute(text(sql2)).scalar()
        print(f"Result: {result2}")

except Exception as e:
    print(f"ERROR: {e}")
