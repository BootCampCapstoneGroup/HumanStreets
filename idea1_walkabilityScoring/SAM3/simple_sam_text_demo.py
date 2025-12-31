import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from ultralytics import SAM, YOLOWorld

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Image to test
IMAGE_PATH = "datasets/sidewalk_segmentation/images/train/tile_6144_13312.jpg"

# 2. The Prompt (What you want to find)
TEXT_PROMPT = "sidewalk" 

# 3. The Models
# Note: SAM models (even new ones) usually expect POINTS or BOXES, not text.
# So we use a 2-step process called "Grounded SAM":
#   Step A: YOLO-World finds the BOX for "sidewalk"
#   Step B: SAM uses that BOX to find the PRECISE MASK.
YOLO_MODEL_PATH = "yolov8s-world.pt" 
SAM_MODEL_PATH  = "sam3.pt"  # <--- Make sure this file exists in your folder!

# ==========================================
# MAIN SCRIPT
# ==========================================
def main():
    print("\n--- STEP 1: LOAD MODELS ---")
    
    # Check if SAM model exists
    if not os.path.exists(SAM_MODEL_PATH):
        print(f"ERROR: Could not find '{SAM_MODEL_PATH}'.")
        print("Please make sure you have downloaded the model and placed it in this folder.")
        return

    # Load Detector (YOLO-World)
    print(f"Loading Text Detector: {YOLO_MODEL_PATH}")
    detector = YOLOWorld(YOLO_MODEL_PATH)
    
    # Load Segmenter (SAM)
    print(f"Loading Segmenter: {SAM_MODEL_PATH}")
    try:
        segmenter = SAM(SAM_MODEL_PATH)
    except Exception as e:
        print(f"Error loading SAM model: {e}")
        return

    # Read Image
    print(f"Reading image: {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Error: Image not found!")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"\n--- STEP 2: FIND '{TEXT_PROMPT}' (Detection) ---")
    # We tell YOLO-World what to look for
    detector.set_classes([TEXT_PROMPT])
    
    # Run detection
    # conf=0.1: We are lenient, we want to find potential candidate areas
    detection_results = detector.predict(img_rgb, conf=0.1, verbose=False)
    
    # Get the bounding boxes (x1, y1, x2, y2)
    boxes = detection_results[0].boxes.xyxy.cpu().numpy()
    
    if len(boxes) == 0:
        print(f"Result: No candidates found for '{TEXT_PROMPT}'.")
        return
    print(f"Result: Found {len(boxes)} boxes matching '{TEXT_PROMPT}'.")

    print(f"\n--- STEP 3: GENERATE MASKS (Segmentation) ---")
    # Now we act as if we "clicked" on these boxes for SAM.
    # SAM fills in the object details inside the box.
    print("Running SAM on detected boxes...")
    sam_results = segmenter(img_rgb, bboxes=boxes, verbose=False)
    
    if sam_results[0].masks is not None:
        masks = sam_results[0].masks.data.cpu().numpy() # Shape: (Num_Masks, H, W)
        print(f"Result: Generated {len(masks)} precise masks.")
        
        # --- VISUALIZATION ---
        print("\n--- STEP 4: VISUALIZE ---")
        plt.figure(figsize=(12, 12))
        plt.imshow(img_rgb)
        
        # Create a Green Overlay for masks
        overlay_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
        for m in masks:
            overlay_mask = np.logical_or(overlay_mask, m.astype(bool))
            
        green_layer = np.zeros_like(img_rgb)
        green_layer[:, :, 1] = 255 # Green
        
        # Alpha Blending (Green where mask is true)
        alpha = np.zeros(img_rgb.shape[:2], dtype=float)
        alpha[overlay_mask] = 0.5 
        
        plt.imshow(green_layer, alpha=alpha[:, :, None])
        
        # Draw the Prompt Boxes (Red)
        ax = plt.gca()
        for box in boxes:
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            
        plt.title(f"Prompt: '{TEXT_PROMPT}'\nRed=YOLO Detection | Green=SAM Segmentation")
        plt.axis('off')
        
        out_file = "sam3_text_result.jpg"
        plt.savefig(out_file)
        print(f"Success! Image saved to: {out_file}")
    else:
        print("SAM returned no masks for these boxes.")

if __name__ == "__main__":
    main()
