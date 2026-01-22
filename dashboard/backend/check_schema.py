
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

# Fallback to default if not set
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:12345@localhost:5432/capstone")

print(f"Connecting to: {DATABASE_URL}")

try:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        print("Checking schema for 'neighborhoods' table:")
        result = conn.execute(text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'neighborhoods';
        """))
        for row in result:
            print(f"  {row[0]}: {row[1]}")
            
except Exception as e:
    print(f"Error: {e}")
