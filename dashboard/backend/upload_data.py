
import os
import pandas as pd
from sqlalchemy import create_engine
import geopandas as gpd

# Configuration
DB_URL = "postgresql://postgres:12345@localhost:5432/capstone"

DATA_DIR = r"v:\MICS\Projects___IN_PROGRESS\DevPorj\BootCamp_Capstone_Project\idea1_walkabilityScoring\cloned\HumanStreets\data"
PARQUET_FILE = os.path.join(DATA_DIR, "riyadh_h3_r9.parquet")
NEIGHBORHOODS_FILE = os.path.join(DATA_DIR, "riyadh_neighborhoods.geojson")
STREETS_FILE = os.path.join(DATA_DIR, "streets.geojson")
SEGMENTS_GPKG = os.path.join(DATA_DIR, "sam3_results_50_overalping.gpkg")

def get_engine():
    return create_engine(DB_URL)

def upload_file(path, table_name, engine):
    if not os.path.exists(path):
        print(f"Skipping {table_name}: File not found at {path}")
        return

    print(f"Uploading {os.path.basename(path)} to table '{table_name}'...")
    try:
        # Determine how to read based on extension
        ext = os.path.splitext(path)[1].lower()
        
        if ext == '.parquet':
            df = pd.read_parquet(path)
            # Ensure index handling if needed
            if 'h3_index' not in df.columns and df.index.name == 'h3_index':
                df = df.reset_index()
            # Parquet usually non-spatial (or handled manually), upload as standard table
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            
        elif ext in ['.geojson', '.gpkg']:
            gdf = gpd.read_file(path)
            # Standardize CRS to 4326
            if gdf.crs != "EPSG:4326":
                print("Reprojecting to EPSG:4326...")
                gdf = gdf.to_crs("EPSG:4326")
            
            # PostGIS upload
            gdf.to_postgis(table_name, engine, if_exists='replace', index=False)
            
        print(f"✅ {table_name} uploaded successfully.")
        
    except Exception as e:
        print(f"❌ Failed to upload {table_name}: {e}")

def main():
    engine = get_engine()
    
    upload_file(PARQUET_FILE, 'h3_grid', engine)
    upload_file(NEIGHBORHOODS_FILE, 'neighborhoods', engine)
    upload_file(STREETS_FILE, 'streets', engine)

if __name__ == "__main__":
    main()
