"""
Enrich h3_grid table with:
1. geometry column (Polygon from H3 index)
2. neighborhood_name_en (via spatial join)
"""
import h3
from shapely.geometry import Polygon
import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "postgresql://postgres:12345@localhost:5432/capstone"

def h3_to_polygon(h3_index):
    """Convert H3 index to Shapely Polygon."""
    try:
        boundary = h3.cell_to_boundary(h3_index)
        # h3 returns (lat, lng), Shapely needs (lng, lat)
        coords = [(lng, lat) for lat, lng in boundary]
        return Polygon(coords)
    except Exception as e:
        print(f"Error converting {h3_index}: {e}")
        return None

def main():
    engine = create_engine(DB_URL)
    
    print("Loading h3_grid from database...")
    h3_df = pd.read_sql("SELECT * FROM h3_grid", engine)
    print(f"Loaded {len(h3_df)} rows.")
    
    print("Converting H3 indexes to geometry...")
    h3_df['geometry'] = h3_df['h3_index'].apply(h3_to_polygon)
    
    # Filter out any failed conversions
    h3_df = h3_df[h3_df['geometry'].notna()]
    print(f"Converted {len(h3_df)} rows successfully.")
    
    # Create GeoDataFrame
    h3_gdf = gpd.GeoDataFrame(h3_df, geometry='geometry', crs="EPSG:4326")
    
    print("Loading neighborhoods from database...")
    neighborhoods_gdf = gpd.read_postgis("SELECT name, geometry FROM neighborhoods", engine, geom_col='geometry')
    print(f"Loaded {len(neighborhoods_gdf)} neighborhoods.")
    
    print("Performing spatial join...")
    # sjoin assigns neighborhood 'name' to each H3 cell based on intersection
    h3_enriched = gpd.sjoin(h3_gdf, neighborhoods_gdf, how='left', predicate='intersects')
    
    # Rename 'name' to 'neighborhood_name_en' and drop 'index_right' if exists
    if 'name' in h3_enriched.columns:
        h3_enriched['neighborhood_name_en'] = h3_enriched['name']
        h3_enriched = h3_enriched.drop(columns=['name', 'index_right'], errors='ignore')
    
    # Drop duplicates (H3 cell may intersect multiple neighborhoods, keep first)
    h3_enriched = h3_enriched.drop_duplicates(subset='h3_index', keep='first')
    
    print(f"Enriched data: {len(h3_enriched)} rows.")
    print(f"Sample: {h3_enriched[['h3_index', 'neighborhood_name_en', 'avg_street_score']].head()}")
    
    print("Uploading enriched h3_grid to PostGIS (replacing old table)...")
    h3_enriched.to_postgis('h3_grid', engine, if_exists='replace', index=False)
    
    print("✅ Enrichment complete!")

if __name__ == "__main__":
    main()
