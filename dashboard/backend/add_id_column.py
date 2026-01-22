
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:12345@localhost:5432/capstone")

try:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        with conn.begin():
            print("Checking if 'neighborhood_id' exists...")
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'neighborhoods' AND column_name = 'neighborhood_id';
            """))
            if not result.fetchone():
                print("Adding 'neighborhood_id' SERIAL PRIMARY KEY...")
                conn.execute(text("ALTER TABLE neighborhoods ADD COLUMN neighborhood_id SERIAL PRIMARY KEY;"))
                print("Column added successfully.")
            else:
                print("'neighborhood_id' already exists.")

except Exception as e:
    print(f"Error: {e}")
