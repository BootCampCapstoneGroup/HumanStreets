import geopandas as gpd
from sqlalchemy import create_engine
from app.core.config import settings

# Setup
engine = create_engine(settings.DATABASE_URL)
sql = "SELECT geometry AS geom, avg_walkability FROM neighborhoods WHERE name ILIKE '%حي النرجس%'"

print(f"Executing: {sql}")

try:
    gdf = gpd.read_postgis(sql, engine, geom_col='geom')
    print(f"Success! Found {len(gdf)} rows")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
