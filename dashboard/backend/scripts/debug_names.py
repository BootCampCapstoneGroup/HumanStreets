import geopandas as gpd
from sqlalchemy import create_engine, text
from app.core.config import settings

engine = create_engine(settings.DATABASE_URL)

print("--- Inspecting Neighborhood Names (searching for *Narjis* pattern) ---")
# Search for anything looking like Narjis (u0627 is Alef, u0644 is Lam, u0646 is Noon...)
# النرجس = Al-Narjis
sql = "SELECT name, avg_walkability FROM neighborhoods WHERE name ILIKE '%نرجس%'"

try:
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        rows = result.fetchall()
        print(f"Found {len(rows)} matching rows:")
        for row in rows:
            print(f" - Name: '{row[0]}' (len={len(row[0])}) | Walkability: {row[1]}")
            # print hex codes to verify hidden chars
            print(f"   Hex: {[hex(ord(c)) for c in row[0]]}")
            
    print("\n--- Inspecting First 10 Neighborhoods ---")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT name FROM neighborhoods LIMIT 10"))
        for row in result:
             print(f" - '{row[0]}'")

except Exception as e:
    print(f"Error: {e}")
