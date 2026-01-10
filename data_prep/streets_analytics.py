
import geopandas as gpd
import pandas as pd
import os
import warnings
from shapely.errors import ShapelyDeprecationWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Correct paths based on user request
GPKG_PATH = os.path.join(BASE_DIR, "..", "data", "sam3_results_50_overalping.gpkg")
STREETS_PATH = os.path.join(BASE_DIR, "..", "data", "streets.geojson")
OUTPUT_PATH = os.path.join(BASE_DIR, "streets_analytics.geojson")

# Layers expected in GPKG
LAYERS = ["road", "sidewalk", "car", "obstacle", "tree"]

# Buffer size in meters
BUFFER_METERS = 30.0

def main():
    print("--- Starting Street Analytics ---")
    
    # 1. Load Streets
    print(f"Loading Streets from {STREETS_PATH}...")
    if not os.path.exists(STREETS_PATH):
        print(f"Error: Streets file not found at {STREETS_PATH}")
        return

    gdf_streets = gpd.read_file(STREETS_PATH)
    print(f"Loaded {len(gdf_streets)} streets.")
    
    # 2. Project Streets (Estimate UTM or use specific)
    # Using EPSG:32618 based on user logs, but ideally we estimate.
    # Let's check current CRS.
    if gdf_streets.crs.to_string() != "EPSG:32618":
        print("Reprojecting streets to EPSG:32618 for buffering...")
        gdf_streets = gdf_streets.to_crs("EPSG:32618")
        
    # 3. Create Street Buffers
    print(f"Buffering streets by {BUFFER_METERS} meters...")
    # We keep the originals for geometry, but use buffers for spatial join
    gdf_buffers = gdf_streets.copy()
    gdf_buffers['geometry'] = gdf_buffers.geometry.buffer(BUFFER_METERS)
    
    # Initialize counts
    for layer in LAYERS:
        gdf_streets[f"count_{layer}"] = 0
        
    # 4. Process Each Layer
    if not os.path.exists(GPKG_PATH):
        print(f"Error: GPKG file not found at {GPKG_PATH}")
        return

    for layer_name in LAYERS:
        print(f"\nProcessing layer: {layer_name}...")
        try:
            # Read layer
            gdf_layer = gpd.read_file(GPKG_PATH, layer=layer_name)
            
            if len(gdf_layer) == 0:
                print(f"  No features in {layer_name}.")
                continue
                
            # Reproject if needed
            if gdf_layer.crs != gdf_streets.crs:
                gdf_layer = gdf_layer.to_crs(gdf_streets.crs)
            
            # Spatial Join: Find segments within street buffers
            # 'inner' join keeps only matches
            # We join segments (left) to buffers (right)
            print(f"  Performing spatial join with {len(gdf_layer)} items...")
            joined = gpd.sjoin(gdf_layer, gdf_buffers, how="inner", predicate="intersects")
            
            # Count by index_right (which corresponds to street index)
            # Assuming streets have a unique index
            counts = joined.groupby("index_right").size()
            
            # Map counts back to street gdf
            # We look up the count for each street index
            gdf_streets[f"count_{layer}"] = counts.reindex(gdf_streets.index, fill_value=0)
            
            print(f"  Total mapped {layer_name}s: {counts.sum()}")
            
        except ValueError as ve:
             print(f"  Warning: Layer '{layer_name}' likely missing in GPKG or error reading: {ve}")
        except Exception as e:
            print(f"  Error processing {layer_name}: {e}")

    # 5. Summarize and Save
    print("\n--- Summary Stats ---")
    print(gdf_streets[[f"count_{l}" for l in LAYERS]].sum())
    
    # Reproject back to WGS84 for GeoJSON
    print("\nReprojecting to WGS84...")
    gdf_final = gdf_streets.to_crs("EPSG:4326")
    
    print(f"Saving to {OUTPUT_PATH}...")
    gdf_final.to_file(OUTPUT_PATH, driver="GeoJSON")
    print("Done!")

if __name__ == "__main__":
    main()
