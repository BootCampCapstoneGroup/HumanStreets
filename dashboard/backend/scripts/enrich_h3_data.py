import sys
import os

# Add parent directory to path to import app.core.config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from sqlalchemy import create_engine, text
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import h3

def h3_to_polygon(h3_index):
    """Convert H3 index to Shapely Polygon."""
    try:
        boundary = h3.cell_to_boundary(h3_index)
        # h3 returns (lat, lng), Shapely needs (lng, lat)
        coords = [(lng, lat) for lat, lng in boundary]
        return Polygon(coords)
    except Exception as e:
        # print(f"Error converting {h3_index}: {e}")
        return None

def main():
    print(f"Connecting to DB: {settings.DATABASE_URL}")
    engine = create_engine(settings.DATABASE_URL)
    
    print("Loading h3_grid from database...")
    try:
        h3_df = pd.read_sql("SELECT * FROM h3_grid", engine)
        print(f"Loaded {len(h3_df)} rows.")
    except Exception as e:
        print(f"Error loading h3_grid: {e}")
        return

    print("Converting H3 indexes to geometry...")
    h3_df['geometry'] = h3_df['h3_index'].apply(h3_to_polygon)
    h3_df = h3_df[h3_df['geometry'].notna()]
    print(f"Converted {len(h3_df)} rows successfully.")
    
    h3_gdf = gpd.GeoDataFrame(h3_df, geometry='geometry', crs="EPSG:4326")
    
    print("Loading neighborhoods from database...")
    neighborhoods_gdf = gpd.read_postgis("SELECT name, geometry FROM neighborhoods", engine, geom_col='geometry')
    print(f"Loaded {len(neighborhoods_gdf)} neighborhoods.")
    
    print("Performing spatial join...")
    h3_enriched = gpd.sjoin(h3_gdf, neighborhoods_gdf, how='left', predicate='intersects')
    
    # Use 'name' from neighborhoods as the grouping key
    if 'name' in h3_enriched.columns:
        h3_enriched['neighborhood_name'] = h3_enriched['name']
    
    print(f"Enriched data: {len(h3_enriched)} rows.")
    
    # Check what score column we have
    score_col = 'avg_street_score'
    if score_col not in h3_enriched.columns:
        print(f"Warning: '{score_col}' not found. Available: {h3_enriched.columns}")
        # Try finding a likely candidate
        candidates = [c for c in h3_enriched.columns if 'score' in c]
        if candidates:
            score_col = candidates[0]
            print(f"Using '{score_col}' instead.")
        else:
            print("No score column found! Aborting.")
            return

    print("Aggregating scores to neighborhoods...")
    neighborhood_scores = h3_enriched.groupby('neighborhood_name')[score_col].mean().reset_index()
    print(f"Calculated scores for {len(neighborhood_scores)} neighborhoods.")
    
    print("Updating 'avg_walkability' in neighborhoods table...")
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE neighborhoods ADD COLUMN IF NOT EXISTS avg_walkability FLOAT"))
        
        # Reset all to NULL first to be sure
        conn.execute(text("UPDATE neighborhoods SET avg_walkability = NULL"))
        
        # Batch update
        updated_count = 0
        for index, row in neighborhood_scores.iterrows():
            name = row['neighborhood_name']
            score = row[score_col]
            
            # Update matching the Arabic Name
            query = text("UPDATE neighborhoods SET avg_walkability = :score WHERE name = :name")
            result = conn.execute(query, {"score": score, "name": name})
            updated_count += result.rowcount
            
        conn.commit()
        print(f"Updated {updated_count} rows in total.")

    # --- VERIFICATION ---
    print("\n--- VERIFICATION ---")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT name, avg_walkability FROM neighborhoods WHERE avg_walkability IS NOT NULL LIMIT 5"))
        rows = result.fetchall()
        print(f"Sample Updated Rows: {len(rows)}")
        for r in rows:
            print(r)
        
        count = conn.execute(text("SELECT COUNT(*) FROM neighborhoods WHERE avg_walkability > 0")).scalar()
        print(f"Total neighborhoods with positive score: {count}")
            
    print("✅ Enrichment and Neighborhood Update complete!")

if __name__ == "__main__":
    main()
