import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt

# 1. Setup Model
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.70,
    stability_score_thresh=0.80,
    crop_n_layers=0,
    min_mask_region_area=10,
)

# 2. Load and Initial Pre-processing
image = cv2.imread('images/backlit/img3.jpg')
height, width = image.shape[:2]
max_dim = 1000
if max(height, width) > max_dim:
    scale = max_dim / max(height, width)
    image = cv2.resize(image, None, fx=scale, fy=scale)

# 3. Robust Dish Detection (Contour-Based)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. Use Adaptive Threshold to handle uneven backlighting
# This is much better than Canny when the rim is faint.
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY_INV, 11, 2
)

# 2. Clean up noise (remove small dots inside/outside)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# 3. Find all contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

dish_mask = np.zeros(image.shape[:2], dtype=np.uint8)
best_contour = None
max_circularity = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    if perimeter == 0 or area < (image.shape[0] * image.shape[1] * 0.1):
        continue  # Skip tiny shapes
    
    # Circularity formula: (4 * pi * Area) / (Perimeter^2)
    # A perfect circle = 1.0. Most petri dishes are > 0.8
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    
    # We want the most circular shape that is also large
    if circularity > max_circularity:
        max_circularity = circularity
        best_contour = cnt

if best_contour is not None:
    # Fit a circle to the best circular contour found
    (x, y), radius = cv2.minEnclosingCircle(best_contour)
    center = (int(x), int(y))
    radius = int(radius)
    
    # Draw mask (shrunk slightly to avoid the rim)
    cv2.circle(dish_mask, center, int(radius * 0.95), 255, thickness=-1)
    circle_data = (center, radius)
    print(f"Dish detected! Circularity: {max_circularity:.2f}")
else:
    print("No circular dish found. Defaulting to center-crop.")
    # Fallback: create a circle in the middle of the image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    radius = int(min(h, w) * 0.45)
    cv2.circle(dish_mask, center, radius, 255, thickness=-1)
    circle_data = (center, radius)

# 4. Enhance Image ONLY inside the dish
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(image_lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
limg = cv2.merge((cl,a,b))
enhanced_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

# Mask the enhanced image so SAM doesn't look outside the dish
masked_enhanced = cv2.bitwise_and(enhanced_rgb, enhanced_rgb, mask=dish_mask)

# 5. Generate and Filter Masks
print("Generating masks...")
masks = mask_generator.generate(masked_enhanced)

filtered_masks = []
for ann in masks:
    # Get center of the mask to check if it's inside our circle
    m_y, m_x = np.where(ann['segmentation'])
    if len(m_x) == 0: continue
    
    cx, cy = np.mean(m_x), np.mean(m_y)
    
    # 1. Filter: Center must be inside the circular mask
    if dish_mask[int(cy), int(cx)] == 0:
        continue
        
    # 2. Filter: Size constraints (ignore huge background fragments)
    if ann['area'] > (image.shape[0] * image.shape[1] * 0.1):
        continue
    
    filtered_masks.append(ann)

print(f"Process Complete. Found {len(filtered_masks)} colonies inside the dish.")

# 6. Visualization
# 5. Visualization Fix for Contour/Hull
plt.figure(figsize=(10,10))
plt.imshow(masked_enhanced)

# Instead of plt.Circle, we plot the coordinates of the hull
if 'hull' in locals() and hull is not None:
    # Convert hull to a closed loop for plotting (N, 1, 2) -> (N, 2)
    hull_pts = hull.reshape(-1, 2)
    # Close the loop by adding the first point to the end
    hull_pts = np.vstack([hull_pts, hull_pts[0]]) 
    plt.plot(hull_pts[:, 0], hull_pts[:, 1], color='red', linewidth=2, linestyle='--', label='Dish Boundary')
    plt.legend()


# Overlay SAM annotations
def show_anns(anns):
    if len(anns) == 0: return
    ax = plt.gca()
    for ann in anns:
        m = ann['segmentation']
        color = np.concatenate([np.random.random(3), [0.5]])
        ax.imshow(np.dstack([np.ones((m.shape[0], m.shape[1], 3)) * color[:3], m * color[3]]))

show_anns(filtered_masks)
plt.axis('off')
plt.title(f"Detected {len(filtered_masks)} Colonies")
plt.show()