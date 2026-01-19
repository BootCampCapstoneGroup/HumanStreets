import sys
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Append path to find app module (if needed, but we import settings)
sys.path.append(os.getcwd())

from app.core.config import settings

def check_names():
    engine = create_engine(settings.DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        # Check for matching names 
        print("Checking for 'Malaz' or 'الملز' variants...")
        query = text("SELECT name FROM neighborhoods WHERE name ILIKE '%Malaz%' OR name ILIKE '%الملز%' OR name ILIKE '%maloz%'")
        res = session.execute(query).fetchall()
        matched = [r[0] for r in res]
        print(f"Found matches: {matched}")
        
        # Print a sample of all names just in case
        print("\nRandom Sample of 20 names:")
        sample = session.execute(text("SELECT name FROM neighborhoods LIMIT 20")).fetchall()
        print([r[0] for r in sample])
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    check_names()
