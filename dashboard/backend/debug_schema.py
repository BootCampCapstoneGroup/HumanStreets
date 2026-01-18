from sqlalchemy import create_engine, text
from app.core.config import settings

def check_schema():
    engine = create_engine(settings.DATABASE_URL)
    with engine.connect() as conn:
        print("--- h3_grid columns ---")
        result = conn.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'h3_grid'"))
        for row in result:
            print(row)
        
        print("\n--- neighborhoods columns ---")
        result = conn.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'neighborhoods'"))
        for row in result:
            print(row)

if __name__ == "__main__":
    check_schema()
