from ultralytics import SAM, YOLOWorld
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Config ---
IMAGE_PATH = "datasets/sidewalk_segmentation/images/train/tile_6144_13312.jpg"
SAM_MODEL = "sam2_t.pt"
YOLO_WORLD_MODEL = "yolov8s-world.pt" # We use this to understand "sidewalk"
TEXT_PROMPT = "sidewalk" 

def main():
    print(f"Loading YOLO-World: {YOLO_WORLD_MODEL}...")
    yolo_model = YOLOWorld(YOLO_WORLD_MODEL)
    
    # Define custom classes (Open Vocabulary detection)
    print(f"Setting text prompt to: '{TEXT_PROMPT}'")
    yolo_model.set_classes([TEXT_PROMPT])
    
    print(f"Loading SAM: {SAM_MODEL}...")
    sam_model = SAM(SAM_MODEL)
    
    print(f"Reading image: {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Error reading image.")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Detect with Text Prompt
    print("Step 1: Detecting objects with YOLO-World...")
    det_results = yolo_model.predict(img_rgb, conf=0.10, verbose=False)
    
    if len(det_results[0].boxes) == 0:
        print("No objects found with that text prompt.")
        return
        
    bboxes = det_results[0].boxes.xyxy.cpu().numpy()
    print(f"Found {len(bboxes)} bounding boxes for '{TEXT_PROMPT}'.")
    
    # 2. Segment with SAM (using boxes as prompts)
    print("Step 2: Segmenting with SAM using bounding boxes...")
    # Ultralytics SAM `predict` accepts `bboxes`
    sam_results = sam_model(img_rgb, bboxes=bboxes, verbose=False)
    
    # 3. Visualize
    if sam_results[0].masks is not None:
        masks = sam_results[0].masks.data.cpu().numpy()
        print(f"Generated {len(masks)} masks.")
        
        plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb)
        
        # Combine all masks
        combined_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
        for m in masks:
            # Resize if needed (SAM masks might be simpler?)
            # Usually strict match.
            combined_mask = np.logical_or(combined_mask, m.astype(bool))
            
        # Overlay
        overlay = np.zeros_like(img_rgb)
        overlay[:, :, 1] = 255 # Green
        
        alpha_mask = np.zeros(img_rgb.shape[:2], dtype=float)
        alpha_mask[combined_mask] = 0.5
        
        plt.imshow(overlay, alpha=alpha_mask[:, :, None])
        
        # Draw boxes too for reference
        ax = plt.gca()
        for box in bboxes:
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            
        plt.title(f"Text Prompt: '{TEXT_PROMPT}' (YOLO derived boxes -> SAM)")
        plt.axis('off')
        plt.savefig("text_prompt_result.jpg")
        print("Saved result to text_prompt_result.jpg")

    else:
        print("SAM returned no masks.")

if __name__ == "__main__":
    main()
