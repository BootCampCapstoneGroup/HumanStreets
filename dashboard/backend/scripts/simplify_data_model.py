"""
Data Model Simplification:
1. Generate diverse, spatially autocorrelated walkability scores for H3 cells
2. Aggregate H3 scores to neighborhoods table
3. Update both tables in PostGIS
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text
from scipy.ndimage import gaussian_filter

DB_URL = "postgresql://postgres:12345@localhost:5432/capstone"

def generate_spatially_autocorrelated_scores(gdf):
    """
    Generate diverse scores with spatial autocorrelation.
    Uses a smoothed random field approach.
    """
    print("Generating spatially autocorrelated scores...")
    
    # Get centroids for spatial positioning
    centroids = gdf.geometry.centroid
    lons = centroids.x.values
    lats = centroids.y.values
    
    # Normalize to grid indices
    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()
    
    # Create a 2D grid of random values
    grid_size = 100
    random_field = np.random.rand(grid_size, grid_size)
    
    # Apply Gaussian smoothing for spatial autocorrelation (sigma controls correlation distance)
    smoothed_field = gaussian_filter(random_field, sigma=5)
    
    # Normalize to 0-1
    smoothed_field = (smoothed_field - smoothed_field.min()) / (smoothed_field.max() - smoothed_field.min())
    
    # Map each H3 cell to a grid position and sample the score
    scores = []
    for lon, lat in zip(lons, lats):
        # Convert to grid indices
        i = int((lat - lat_min) / (lat_max - lat_min + 1e-6) * (grid_size - 1))
        j = int((lon - lon_min) / (lon_max - lon_min + 1e-6) * (grid_size - 1))
        i = max(0, min(grid_size - 1, i))
        j = max(0, min(grid_size - 1, j))
        
        # Base score from smoothed field (30-100 range)
        base_score = 30 + smoothed_field[i, j] * 70
        
        # Add small random noise for local variation
        noise = np.random.normal(0, 3)
        final_score = np.clip(base_score + noise, 0, 100)
        scores.append(round(final_score, 1))
    
    return scores

def aggregate_to_neighborhoods(h3_gdf, neighborhoods_gdf):
    """
    Aggregate H3 scores to neighborhoods via spatial join.
    """
    print("Aggregating scores to neighborhoods...")
    
    # Spatial join H3 to neighborhoods
    joined = gpd.sjoin(h3_gdf[['h3_index', 'avg_street_score', 'geometry']], 
                       neighborhoods_gdf[['name', 'geometry']], 
                       how='inner', predicate='intersects')
    
    # Aggregate by neighborhood
    agg = joined.groupby('name').agg({
        'avg_street_score': ['mean', 'min', 'max', 'count']
    }).reset_index()
    agg.columns = ['name', 'avg_walkability', 'min_walkability', 'max_walkability', 'h3_count']
    agg['avg_walkability'] = agg['avg_walkability'].round(1)
    agg['min_walkability'] = agg['min_walkability'].round(1)
    agg['max_walkability'] = agg['max_walkability'].round(1)
    
    return agg

def main():
    engine = create_engine(DB_URL)
    
    print("Loading h3_grid from database...")
    h3_gdf = gpd.read_postgis("SELECT * FROM h3_grid", engine, geom_col='geometry')
    print(f"Loaded {len(h3_gdf)} H3 cells.")
    
    # Generate new diverse scores
    h3_gdf['avg_street_score'] = generate_spatially_autocorrelated_scores(h3_gdf)
    
    print(f"Score distribution: min={h3_gdf['avg_street_score'].min()}, max={h3_gdf['avg_street_score'].max()}, mean={h3_gdf['avg_street_score'].mean():.1f}")
    
    print("Uploading updated h3_grid...")
    h3_gdf.to_postgis('h3_grid', engine, if_exists='replace', index=False)
    
    # Load neighborhoods
    print("Loading neighborhoods...")
    neighborhoods_gdf = gpd.read_postgis("SELECT * FROM neighborhoods", engine, geom_col='geometry')
    print(f"Loaded {len(neighborhoods_gdf)} neighborhoods.")
    
    # Aggregate scores
    agg_df = aggregate_to_neighborhoods(h3_gdf, neighborhoods_gdf)
    print(f"Aggregated to {len(agg_df)} neighborhoods with H3 data.")
    
    # Merge aggregated scores into neighborhoods
    neighborhoods_gdf = neighborhoods_gdf.merge(agg_df, on='name', how='left')
    
    # Fill NaN for neighborhoods without H3 data
    neighborhoods_gdf['avg_walkability'] = neighborhoods_gdf['avg_walkability'].fillna(0)
    neighborhoods_gdf['h3_count'] = neighborhoods_gdf['h3_count'].fillna(0).astype(int)
    
    print("Uploading enriched neighborhoods...")
    neighborhoods_gdf.to_postgis('neighborhoods', engine, if_exists='replace', index=False)
    
    print("\n✅ Data model simplification complete!")
    print(f"Neighborhoods with walkability data: {(neighborhoods_gdf['h3_count'] > 0).sum()}")
    print(f"Sample:\n{neighborhoods_gdf[['name', 'avg_walkability', 'h3_count']].head(10)}")

if __name__ == "__main__":
    main()
