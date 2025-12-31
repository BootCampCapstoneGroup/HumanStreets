import folium
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import shape

# --- 1. Initialize Models ---
# Use YOLO-World for text-based detection
from ultralytics import YOLOWorld, SAM
detection_model = YOLOWorld("yolov8s-world.pt")  # Efficient open-vocabulary detector
detection_model.set_classes(["road", "pavement", "path"])

# Use SAM 2 for refinement/segmentation
segmentation_model = SAM("sam2_b.pt")

from rasterio.windows import Window
from itertools import product

# --- 2. Run Segmentation (Tiled) ---
image_path = "./mosiac_rgb_6cmPerPixel.tif"
tile_size = 1024  # Process in 1024x1024 chunks

with rasterio.open(image_path) as src:
    H, W = src.height, src.width
    profile = src.profile
    global_mask = np.zeros((H, W), dtype=np.uint8)
    
    # Create windows
    # We iterate with some overlap if needed, but for now non-overlapping tiles is safer for memory/speed
    # unless boundary artifacts are a major concern. Let's start with non-overlapping.
    for col_off, row_off in product(range(0, W, tile_size), range(0, H, tile_size)):
        window = Window(col_off, row_off, min(tile_size, W - col_off), min(tile_size, H - row_off))
        
        # Read tile
        tile_data = src.read(window=window)
        # Reshape to (H, W, C) for Ultralytics: (C, H, W) -> (H, W, C)
        tile_img = np.moveaxis(tile_data, 0, -1)
        
        # Skip empty tiles (optional optimization)
        if tile_img.max() == 0:
            continue
            
        print(f"Processing tile: {window}")

        # Step 1: Detect sidewalks in tile
        # conf=0.1 lower threshold to catch potential sidewalks
        det_results = detection_model.predict(tile_img, conf=0.05, verbose=False) 
        
        # Boxes are relative to the tile (0,0)
        bboxes = det_results[0].boxes.xyxy.cpu().numpy()
        
        if len(bboxes) > 0:
            # Step 2: Segment using SAM 2
            # SAM needs the image and the boxes
            results = segmentation_model(tile_img, bboxes=bboxes, verbose=False)
            
            tile_mask_accum = np.zeros((window.height, window.width), dtype=np.uint8)
            
            for result in results:
                if result.masks is not None:
                     # Combine all masks in this result
                     m = result.masks.data.cpu().numpy().astype('uint8')
                     # m shape is (N, H, W), take max across N
                     if m.shape[0] > 0:
                        m_max = np.max(m, axis=0)
                        tile_mask_accum = np.maximum(tile_mask_accum, m_max)

            # Place into global mask
            # global_mask is (H, W)
            # window slices: row_off:row_off+h, col_off:col_off+w
            global_mask[row_off:row_off+window.height, col_off:col_off+window.width] = tile_mask_accum

# Local mask variable for subsequent code compatibility
mask = global_mask 

# --- 3. Convert Mask to GeoJSON ---
with rasterio.open(image_path) as src:
    transform = src.transform
    crs = src.crs

# Extract shapes (polygons) from the binary mask
mask_shapes = features.shapes(mask, transform=transform)

# Filter for the "1" values (the sidewalk area)
polygons = []
for geom, value in mask_shapes:
    if value == 1:
        # Convert to a format Folium understands (Lat/Lon)
        s = shape(geom)
        polygons.append(s)

# --- 4. Create Folium Map ---
# Centered on Riyadh
m = folium.Map(location=[24.7136, 46.6753], zoom_start=17, tiles='CartoDB positron')

# Add the segmented sidewalks as a layer
for poly in polygons:
    # Folium expects [Lat, Lon], but GeoTIFFs often use [Lon, Lat]
    # We swap coordinates if necessary
    geo_j = folium.GeoJson(
        poly.__geo_interface__,
        style_function=lambda x: {'fillColor': '#00ff00', 'color': '#00ff00', 'weight': 2, 'fillOpacity': 0.5}
    )
    geo_j.add_to(m)

# Save and Show
m.save("riyadh_sidewalk_analysis.html")
print("Map saved to riyadh_sidewalk_analysis.html. Open this file in your browser.")