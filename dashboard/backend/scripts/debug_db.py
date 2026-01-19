from sqlalchemy import create_engine, text
from app.core.config import settings

def check_data():
    engine = create_engine(settings.DATABASE_URL)
    with engine.connect() as conn:
        print("Checking h3_grid table...")
        try:
            result = conn.execute(text("SELECT COUNT(*) FROM h3_grid"))
            count = result.scalar()
            print(f"Total rows in h3_grid: {count}")
            
            if count > 0:
                print("Sample Data:")
                result = conn.execute(text("SELECT h3_index, neighborhood_name_en, avg_street_score FROM h3_grid LIMIT 5"))
                for row in result:
                    print(row)
            else:
                print("Table 'h3_grid' is empty!")
                
        except Exception as e:
            print(f"Error querying h3_grid: {e}")

        print("\nChecking neighborhoods table...")
        try:
            result = conn.execute(text("SELECT COUNT(*) FROM neighborhoods"))
            count = result.scalar()
            print(f"Total rows in neighborhoods: {count}")
            if count > 0:
                print("Sample Neighborhood:")
                result = conn.execute(text("SELECT name_en FROM neighborhoods LIMIT 1"))
                for row in result:
                    print(row)
        except Exception as e:
            print(f"Error checking neighborhoods: {e}")

if __name__ == "__main__":
    check_data()
