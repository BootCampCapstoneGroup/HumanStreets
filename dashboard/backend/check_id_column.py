
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:12345@localhost:5432/capstone")

try:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        print("Checking ALL columns for 'neighborhoods' table:")
        result = conn.execute(text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'neighborhoods';
        """))
        columns = result.fetchall()
        for row in columns:
            print(f"  {row[0]}: {row[1]}")
            
        print("\nChecking for Primary Key:")
        pk_result = conn.execute(text("""
            SELECT a.attname
            FROM   pg_index i
            JOIN   pg_attribute a ON a.attrelid = i.indrelid
                                AND a.attnum = ANY(i.indkey)
            WHERE  i.indrelid = 'neighborhoods'::regclass
            AND    i.indisprimary;
        """))
        pk = pk_result.fetchone()
        if pk:
            print(f"  Primary Key: {pk[0]}")
        else:
            print("  No Primary Key found.")

except Exception as e:
    print(f"Error: {e}")
