import os
import cv2
import numpy as np
from ultralytics import YOLO
from matplotlib import colors

# Foreground type and mask color map definition
color_map = {
    0: "red",          # 0 and 1 are both no parking signs
    1: "red",          # No parking signs: red
    2: "deepskyblue",  # Parking line: deepskyblue
    3: "limegreen"     # Key vehicle: limegreen
}

# Convert color name to OpenCV's BGR format (integer value)
def color_name_to_bgr(color_name):
    rgb = colors.to_rgb(color_name)  
    bgr = tuple(int(c * 255) for c in rgb[::-1]) 
    return bgr

color_map_bgr = {cls: color_name_to_bgr(color_name) for cls, color_name in color_map.items()}

# Loading instance segmentation model and parameters
model = YOLO("yolov11/v11n/best.pt")

# Input and output paths
# Foreground consistent image: fci
source = 'mvipd_dataset/valid/images'
output_folder = 'features/yolov11n_fci'
os.makedirs(output_folder, exist_ok=True)

# Traverse the images in the source folder and perform foreground-aware preprocessing
for image_name in os.listdir(source):
    image_path = os.path.join(source, image_name)
    if not os.path.isfile(image_path):
        continue
    
    img = cv2.imread(image_path)
    if img is None:
        #print(f"Unable to read image: {image_path}")
        continue
    
    height, width, _ = img.shape
    black_canvas = np.zeros_like(img)  # Initialize the image to be completely black

    # Using the segmentation model to make predictions
    results = model(image_path)
    result = results[0]
    boxes = result.boxes
    masks = result.masks
    
    if masks is None or boxes is None:
        #print(f"No mask was detected in image {image_name}")
        continue
    
    box_classes = boxes.cls.cpu().numpy()  # Foreground instance category
    xy_coords = masks.xy  # Get the coordinates of each instance
    
    # Create a list to store information about candidate masks with box_cls == 3
    candidate_masks = []
    for mask_idx, (cls, coords) in enumerate(zip(box_classes, xy_coords)):
        cls = int(cls)  
        if cls == 3:
            
            coords = np.array(coords)  
            if coords.size == 0:  
                #print(f"Warning: Empty coords detected, skipping mask. File: {image_name}, Class: {cls}")
                continue            
            min_x, min_y = np.min(coords, axis=0)
            max_x, max_y = np.max(coords, axis=0)
            W1 = (max_x - min_x) + (max_y - min_y)
            candidate_masks.append({
                "mask_idx": mask_idx,
                "coords": coords,
                "min_x": min_x,
                "min_y": min_y,
                "max_x": max_x,
                "max_y": max_y,
                "W1": W1
            })
        else:
            # For the areas where other foreground categories are located, fill them with color directly
            try:
                coords = np.array(coords, dtype=np.int32)  
                if coords.ndim != 2 or coords.shape[1] != 2:
                    #print(f"Error: Invalid coords format, skipping mask. File: {image_name}, Class: {cls}")
                    continue
                
                color = color_map_bgr.get(cls, None)  
                if color is None:
                    #print(f"Error: No colormap found for category {cls}. Skipping this mask! File: {image_name}")
                    continue
                
                cv2.fillPoly(black_canvas, [coords], color)
            except Exception as e:
                # print(f"cv2.fillPoly Error! File: {image_name}, class: {cls}, coords: {coords.shape}, color: {color}")
                # print(f"Error message: {e}")
                continue
 
    # When the instance is a vehicle, select the key vehicle
    if candidate_masks:
        # Size metric
        candidate_masks = sorted(candidate_masks, key=lambda x: x["W1"], reverse=True)[:5]
        W1_max = candidate_masks[0]["W1"]
        for mask_info in candidate_masks:
            mask_info["W1_normalized"] = mask_info["W1"] / W1_max  # 归一化 W1

        # Centricity metric
        image_center = np.array([width / 2, height / 2])
        for mask_info in candidate_masks:
            mask_center = np.array([
                (mask_info["min_x"] + mask_info["max_x"]) / 2,
                (mask_info["min_y"] + mask_info["max_y"]) / 2
            ])
            distance = np.linalg.norm(mask_center - image_center)
            mask_info["W2"] = distance
        W2_min = min(mask_info["W2"] for mask_info in candidate_masks)
        for mask_info in candidate_masks:
            mask_info["W2_normalized"] = W2_min / mask_info["W2"]  # 归一化 W2

        # Calculate the final weight and select the mask with the largest weight
        for mask_info in candidate_masks:
            mask_info["final_weight"] = mask_info["W1_normalized"] * mask_info["W2_normalized"]
        best_mask = max(candidate_masks, key=lambda x: x["final_weight"])

        # Fill the mask with the largest weight
        best_coords = np.array(best_mask["coords"], dtype=np.int32)
        cv2.fillPoly(black_canvas, [best_coords], color_map_bgr[3])

    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, black_canvas)
