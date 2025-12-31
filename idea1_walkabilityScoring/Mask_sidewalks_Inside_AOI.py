import os
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import xy
from itertools import product
from ultralytics import SAM
import matplotlib.pyplot as plt
import random
import yaml
from pathlib import Path
import folium
from shapely.geometry import Polygon as ShapelyPolygon
import json
from pyproj import Transformer
import base64
from io import BytesIO

# --- Configuration ---
IMAGE_PATH = "./mosiac_rgb_6cmPerPixel.tif"
TILE_SIZE = 1024
OUTPUT_DIR = "datasets/sidewalk_segmentation"
TRAIN_RATIO = 0.90
VISUALIZATION_SAMPLES = 5 

# Specific tiles to debug (col_off, row_off)
DEBUG_TILES = [
    (6144, 13312),
    (1024, 19456), 
    (1024, 20480)
]
FORCE_DEBUG_ONLY = False # ENABLE FULL RUN


# Ensure reproducibility
random.seed(42)

def create_directory_structure():
    """Creates the YOLO dataset directory structure."""
    for split in ["train", "val"]:
        os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)
    print(f"Created dataset structure in {OUTPUT_DIR}")

def normalize_polygon(polygon, width, height):
    """
    Normalizes polygon coordinates (0-1) for YOLO format.
    Polygon shape: (N, 2) where columns are x, y.
    """
    normalized = polygon.astype(float)
    normalized[:, 0] /= width
    normalized[:, 1] /= height
    return normalized.flatten()

def save_sample_visualization(samples, filename="sample_masks.png"):
    """
    Saves a plot of sample images with their masks overlayed.
    samples: List of (image, mask) tuples.
    """
    if not samples:
        return
    
    count = len(samples)
    fig, axes = plt.subplots(1, count, figsize=(5 * count, 5))
    if count == 1:
        axes = [axes]
    
    for ax, (img, mask) in zip(axes, samples):
        ax.imshow(img)
        # Create a colored overlay for the mask
        overlay = np.zeros_like(img)
        overlay[mask == 1] = [0, 255, 0]  # Green for mask
        ax.imshow(overlay, alpha=0.4)
        ax.axis('off')
        ax.set_title("Generated Mask")
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved visualization sample to {filename}")

def main():
    # 1. Setup
    create_directory_structure()
    
    # Initialize models
    print("Loading models...")
    # detection_model = YOLOWorld("yolov8s-world.pt") # Unused
    # detection_model.set_classes(CLASSES) # Unused
    segmentation_model = SAM("sam2_t.pt")
    
    # Store all found polygons for the map
    all_map_polygons = []
    map_overlays = [] # List of (base64_img, bounds)

    # 2. Process Image
    with rasterio.open(IMAGE_PATH) as src:
        W, H = src.width, src.height
        transform = src.transform
        crs = src.crs
        print(f"Image Size: {W}x{H}, Tile Size: {TILE_SIZE}")
        print(f"CRS: {crs}")
        
        # Reprojection Transformer
        transformer = None
        if crs.to_string() != "EPSG:4326":
            print("CRS is not EPSG:4326. Will reproject map coords.")
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True) # returns (lon, lat)
            
        # DEBUG: Print Center
        x_c, y_c = src.xy(H//2, W//2)
        if transformer:
             lon_c, lat_c = transformer.transform(x_c, y_c)
             print(f"Center Image Coords: {int(x_c)}, {int(y_c)} -> Lat/Lon: {lat_c}, {lon_c}")
        else:
             print(f"Center Image Coords (Lat/Lon): {y_c}, {x_c}")
        
        # Iterate over tiles
        if FORCE_DEBUG_ONLY:
            print(f"DEBUG MODE: Processing only {len(DEBUG_TILES)} specific tiles...")
            tile_indices = DEBUG_TILES
        else:
            # Sample 10% but INCLUDE debug tiles to ensure we get some results
            all_indices = list(product(range(0, W, TILE_SIZE), range(0, H, TILE_SIZE)))
            all_indices_set = set(all_indices)
            
            # Remove debug tiles from pool to avoid dups
            for dt in DEBUG_TILES:
                if dt in all_indices_set:
                    all_indices_set.remove(dt)
            
            sample_size = max(1, int(len(all_indices) * 0.10))
            # Ensure sample_size does not exceed the number of available non-debug tiles
            sample_size = min(sample_size, len(all_indices_set))
            random_sample = random.sample(list(all_indices_set), sample_size)
            
            # Limit to 10 tiles total for rapid verify
            MAX_TILES = 10
            
            # Combine
            tile_indices = DEBUG_TILES + random_sample
            tile_indices = tile_indices[:MAX_TILES] # Hard clip
            print(f"Sampling LIMITED to {len(tile_indices)} tiles (User Request)")
        
        processed_count = 0
        
        for min_processed, (col_off, row_off) in enumerate(tile_indices):
            window = Window(col_off, row_off, min(TILE_SIZE, W - col_off), min(TILE_SIZE, H - row_off))
            
            # SKIP if already done
            base_filename = f"tile_{col_off}_{row_off}"
            # Check train/val headers
            exists = False
            for split in ['train', 'val']:
                if os.path.exists(f"{OUTPUT_DIR}/labels/{split}/{base_filename}.txt"):
                    exists = True
                    break
            
            if exists:
                print(f"Skipping {base_filename} (Already Exists)")
                # If we skip, we still might want to add to map if we could read it?
                # For now, simplistic approach: just skip processing. 
                # To ensure map is generated, we might need to read the file? 
                # Let's assume user wants to process 10 NEW or 10 TOTAL?
                # "10 images only, and it has a mask consider it done and skip it"
                # If we skip, we won't add to `all_map_polygons` unless we load it back.
                # So verify: if we skip, we want the map? Yes.
                # Let's simple-load lines back for the map.
                # (Reading back logic)
                for split in ['train', 'val']:
                     p = f"{OUTPUT_DIR}/labels/{split}/{base_filename}.txt"
                     if os.path.exists(p):
                         with open(p, 'r') as f:
                             lines = f.readlines()
                         for line in lines:
                             parts = list(map(float, line.strip().split()[1:]))
                             # Denormalize
                             poly_pts = np.array(parts).reshape(-1, 2)
                             poly_pts[:, 0] *= window.width
                             poly_pts[:, 1] *= window.height
                             
                             global_pixels = poly_pts + [col_off, row_off]
                             rows = global_pixels[:, 1]
                             cols = global_pixels[:, 0]
                             xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
                             if transformer:
                                 lons, lats = transformer.transform(xs, ys)
                                 all_map_polygons.append(list(zip(lats, lons)))
                             else:
                                 all_map_polygons.append(list(zip(ys, xs)))
                 
                # Load image for map overlay (if user wants to see it)
                # Try to find the image file
                img_path = None
                for split in ['train', 'val']:
                    p = f"{OUTPUT_DIR}/images/{split}/{base_filename}.jpg"
                    if os.path.exists(p):
                        img_path = p
                        break
                
                if img_path:
                    with open(img_path, "rb") as img_f:
                        b64_data = base64.b64encode(img_f.read()).decode('utf-8')
                        img_src = f"data:image/jpeg;base64,{b64_data}"
                        
                        # Calculate Bounds
                        # window is defined at top of loop
                        l, b, r, t = src.window_bounds(window)
                        if transformer:
                            lon_min, lat_min = transformer.transform(l, b)
                            lon_max, lat_max = transformer.transform(r, t)
                            # Folium bounds: [[lat_min, lon_min], [lat_max, lon_max]]
                            bounds = [[lat_min, lon_min], [lat_max, lon_max]]
                            map_overlays.append((img_src, bounds))
                
                continue


            # Print progress every tile to confirm liveness
            print(f"Scanning tile {min_processed+1}/{len(tile_indices)}: {col_off}_{row_off}...", end="\r", flush=True)

            window = Window(col_off, row_off, min(TILE_SIZE, W - col_off), min(TILE_SIZE, H - row_off))
            
            # Read tile
            tile_data = src.read(window=window)
            if tile_data.shape[0] == 3: # Check for 3 channels (RGB)
                 tile_img = np.moveaxis(tile_data, 0, -1) # (C, H, W) -> (H, W, C)
            else:
                 # Handle cases with alpha channel or other formats if necessary, assuming first 3 are RGB
                 tile_img = np.moveaxis(tile_data[:3, :, :], 0, -1)

            # Skip empty/black tiles
            if tile_img.max() == 0:
                continue
            
            # Ensure uint8
            if tile_img.dtype != np.uint8:
                 if tile_img.max() > 255:
                     tile_img = (tile_img / tile_img.max() * 255).astype(np.uint8)
                 else:
                     tile_img = tile_img.astype(np.uint8)

            # Need valid RGB
            if tile_img.shape[2] != 3:
                 continue
                 
            # -- NEW PIPELINE: SAM Auto-Segment + CLIP Filter -- 
            
            # 1. Run SAM in Auto Mode (No prompts)
            import time
            t0 = time.time()
            sam_results = segmentation_model(tile_img, verbose=False)
            t1 = time.time()
            print(f"SAM took {t1-t0:.2f}s for {col_off}_{row_off}")
            
            final_mask = np.zeros((window.height, window.width), dtype=np.uint8)
            labels = [] # YOLO labels
            
            # Prepare CLIP
            import torch
            import clip
            from PIL import Image
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if 'clip_model' not in globals():
                global clip_model, clip_preprocess
                print("Loading CLIP model...")
                clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
            
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in ["sidewalk", "pavement", "walkway", "ground", "road", "building", "dirt"]]).to(device)
            positive_indices = [0, 1, 2]
            
            if len(sam_results) > 0 and sam_results[0].masks is not None:
                masks = sam_results[0].masks.data.cpu().numpy().astype('uint8') # (N, H, W)
                
                # Check for SAM issue where it returns massive number of masks
                if len(masks) > 300: 
                    # Optimization: Likely too cluttered or noise, skip or limit?
                    # For now proceed but it might be slow.
                    pass 

                for i, m in enumerate(masks):
                    if m.sum() < 100: continue

                    y_indices, x_indices = np.where(m)
                    y_min, y_max = y_indices.min(), y_indices.max()
                    x_min, x_max = x_indices.min(), x_indices.max()
                    
                    h_crop, w_crop = y_max - y_min, x_max - x_min
                    if h_crop < 10 or w_crop < 10: continue
                    
                    crop = tile_img[y_min:y_max+1, x_min:x_max+1]
                    pil_img = Image.fromarray(crop)
                    image_input = clip_preprocess(pil_img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        image_features = clip_model.encode_image(image_input)
                        text_features = clip_model.encode_text(text_inputs)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                        probs = similarity[0].cpu().numpy()
                    
                    top_class_idx = probs.argmax()
                    score = probs[top_class_idx]
                    
                    if top_class_idx in positive_indices and score > 0.3: # Lowered to 0.3
                        final_mask = np.maximum(final_mask, m)
                         
                        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                             if contour.size >= 6:
                                 # 1. YOLO Format (Normalized relative to tile)
                                 norm_poly = normalize_polygon(contour.reshape(-1, 2), window.width, window.height)
                                 labels.append(f"0 {' '.join(map(str, norm_poly))}")
                                 
                                 # 2. Map Format (Global Lat/Lon)
                                 # Contour points: (x, y) relative to tile
                                 # Global pixel: (col_off + x, row_off + y)
                                 global_pixels = contour.reshape(-1, 2) + [col_off, row_off]
                                 
                                 # Rasterio transform expects (rows, cols) -> (y, x) for pixels?
                                 # transform * (col, row) -> (x, y) [Projected Coords]
                                 # src.xy(row, col) -> (x, y)
                                 
                                 # Convert all points
                                 # Note: src.xy takes list of rows, list of cols
                                 rows = global_pixels[:, 1]
                                 cols = global_pixels[:, 0]
                                 xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
                                 
                                 # Project to Lat/Lon for Map
                                 if transformer:
                                     lons, lats = transformer.transform(xs, ys)
                                     all_map_polygons.append(list(zip(lats, lons)))
                                 else:
                                     all_map_polygons.append(list(zip(ys, xs)))


            # 4. Save Data (Only if labels found)
            if len(labels) > 0:
                split = "train" if random.random() < TRAIN_RATIO else "val"
                base_filename = f"tile_{col_off}_{row_off}"
                img_path = f"{OUTPUT_DIR}/images/{split}/{base_filename}.jpg"
                lbl_path = f"{OUTPUT_DIR}/labels/{split}/{base_filename}.txt"
                
                cv2.imwrite(img_path, cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR))
                with open(lbl_path, "w") as f:
                    f.write("\n".join(labels))
                
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} tiles...")

                # Add to map overlays
                pil_img = Image.fromarray(tile_img)
                buff = BytesIO()
                pil_img.save(buff, format="JPEG")
                b64_data = base64.b64encode(buff.getvalue()).decode('utf-8')
                img_src = f"data:image/jpeg;base64,{b64_data}"
                
                l, b, r, t = src.window_bounds(window)
                if transformer:
                     lon_min, lat_min = transformer.transform(l, b)
                     lon_max, lat_max = transformer.transform(r, t)
                     bounds = [[lat_min, lon_min], [lat_max, lon_max]]
                     map_overlays.append((img_src, bounds))

    # 5. Finalize
    # save_sample_visualization(visualization_data) # Skip sample viz for full run or keep it? Keep simple.
    
    # Create dataset.yaml
    yaml_content = {'path': os.path.abspath(OUTPUT_DIR), 'train': 'images/train', 'val': 'images/val', 'names': {0: 'sidewalk'}}
    with open(f"{OUTPUT_DIR}/dataset.yaml", 'w') as f:
        yaml.dump(yaml_content, f)
        
    print(f"\nDataset generation complete! Processed {processed_count} tiles.")
    
    # --- Generate Map ---
    # --- Generate Map ---
    print(f"Generating map with {len(all_map_polygons)} segments...")

    import requests

    def fetch_osm_streets(min_lat, min_lon, max_lat, max_lon):
        """Fetch street data (Highways) from Overpass API."""
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"""
            [out:json];
            (
              way["highway"]({min_lat},{min_lon},{max_lat},{max_lon});
            );
            out geom;
        """
        try:
            response = requests.get(overpass_url, params={'data': overpass_query}, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Overpass API error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Overpass fetch failed: {e}")
            return None

    if len(all_map_polygons) > 0 or transformer:
        # Determine map center and bounds
        if len(all_map_polygons) > 0:
             center = all_map_polygons[0][0]
        elif transformer:
             # Calculate center from image center if no polygons found yet
             # (Reuse vars from earlier print if scope allows, or recalc)
             with rasterio.open(IMAGE_PATH) as s:
                 tx, ty = s.xy(s.height // 2, s.width // 2)
                 t_lon, t_lat = transformer.transform(tx, ty)
                 center = [t_lat, t_lon]
        else:
             center = [0, 0] # Fallback
             
        m = folium.Map(location=center, zoom_start=18, tiles='CartoDB positron')
        
        # Add Sidewalks (Green)
        for poly_points in all_map_polygons:
            folium.Polygon(
                locations=poly_points,
                color="#00ff00",
                weight=2,
                fill_opacity=0.4,
                popup="Sidewalk"
            ).add_to(m)

        # Add Image Overlays
        print(f"Adding {len(map_overlays)} image overlays...")
        for img_src, bounds in map_overlays:
            folium.raster_layers.ImageOverlay(
                image=img_src,
                bounds=bounds,
                opacity=0.6,
                name="Satellite Imagery"
            ).add_to(m)

        # Add OSM Streets (Red)
        if transformer:
             # Calculate bounds for Overpass
             with rasterio.open(IMAGE_PATH) as s:
                 l, b, r, t = s.bounds
                 # Reproject bounds
                 # transform(x, y) -> (lon, lat)
                 lon_min, lat_min = transformer.transform(l, b)
                 lon_max, lat_max = transformer.transform(r, t)
                 
                 # Ensure min/max correct
                 min_lat, max_lat = min(lat_min, lat_max), max(lat_min, lat_max)
                 min_lon, max_lon = min(lon_min, lon_max), max(lon_min, lon_max)
                 
                 print(f"Fetching OSM Strees for bounds: {min_lat:.5f}, {min_lon:.5f} to {max_lat:.5f}, {max_lon:.5f}")
                 osm_data = fetch_osm_streets(min_lat, min_lon, max_lat, max_lon)
                 
                 if osm_data and 'elements' in osm_data:
                     print(f"Found {len(osm_data['elements'])} street segments.")
                     for element in osm_data['elements']:
                         if element['type'] == 'way' and 'geometry' in element:
                             # Extract points (lat, lon)
                             line_points = [[pt['lat'], pt['lon']] for pt in element['geometry']]
                             folium.PolyLine(
                                 line_points, 
                                 color="red", 
                                 weight=2, 
                                 opacity=0.7,
                                 popup=f"Street: {element.get('tags', {}).get('name', 'unnamed')}"
                             ).add_to(m)

        # Add base layers
        folium.TileLayer('openstreetmap').add_to(m)
        folium.LayerControl().add_to(m)

        map_path = "sidewalk_map_full.html"
        m.save(map_path)
        print(f"Map saved to {map_path}")
    else:
        print("No segments found and no transformer setup. Cannot generate valid map.")


if __name__ == "__main__":
    main()
