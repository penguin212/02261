import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import csv

# 1. Setup Model (Use 'vit_b' for speed)
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {device}...")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# 2. Optimized Configuration
# We reduce points_per_side to 32 (standard) but lower the thresholds.
# Crucially, we disable 'crop_n_layers' which is the main cause of hanging.
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,            # Back to 32 for speed
    pred_iou_thresh=0.70,          # Keep low to catch faint objects
    stability_score_thresh=0.80,   # Keep low for consistency
    crop_n_layers=0,               # DISABLE cropping to fix the hang
    min_mask_region_area=10,       # Ignore single-pixel noise
    points_per_batch=64,           # Process in batches to save memory
)

# 3. Smart Pre-processing (The "Secret Sauce")
image = cv2.imread('images/white_bg/img3.jpg')

# Resize if the image is massive (e.g., > 2000px wide) to speed up inference
# SAM works best around 1024-1500px


output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



# 2. Light Blur (Crucial for adaptive thresholding to ignore pixel-level noise)
blurred = cv2.medianBlur(gray, 5)

# 3. Detect the dish to create a mask (to avoid detecting shadows/rims)
mask = np.zeros_like(gray)
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 100, 
                           param1=50, param2=30, minRadius=400, maxRadius=600)


max_x = 0
min_x = 0
max_y = 0
min_y = 0
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles[:1]:
        cv2.circle(mask, (x, y), r, 1, -1) # Slightly smaller to avoid edge
        max_x = x + r
        min_x = x - r
        max_y = y + r
        min_y = y - r
        



image = image * cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# cv2.imshow("img", image)
# cv2.waitKey(0)

image = image[min_y:(max_y + 1), min_x:(max_x + 1)]



# height, width = image.shape[:2]
# max_dim = 1000
# if max(height, width) > max_dim:
#     scale = max_dim / max(height, width)
#     image = cv2.resize(image, None, fx=scale, fy=scale)
print(f"Resized image to {image.shape[:2]} for speed.")



print("Generating masks (this might still take 30-60s on CPU)...")
masks = mask_generator.generate(image)

# 4. Filter Results (Ignore the dish itself)
filtered_masks = []
img_area = image.shape[0] * image.shape[1]
sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)

# Assume largest mask is the dish/background if it's huge
if sorted_masks and sorted_masks[0]['area'] > img_area * 0.5:
    # Use the largest mask to define the valid area
    dish_mask = sorted_masks[0]['segmentation']
    # Invert it if the dish was detected as the hole, or keep it if it's the dish
    # For safety, we usually just skip the very largest mask
    sorted_masks = sorted_masks[1:]

for ann in sorted_masks:
    # Filter out huge background segments
    if ann['area'] > (img_area * 0.2): 
        continue
    # Filter out tiny noise
    if ann['area'] < 10: 
        continue
    filtered_masks.append(ann)

print(f"Process Complete. Found {len(filtered_masks)} colonies.")

# 5. Visualization
def show_anns(anns):
    if len(anns) == 0: return
    ax = plt.gca()
    ax.set_autoscale_on(False)
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.45)))

plt.figure(figsize=(10,10))
plt.imshow(image)
show_anns(filtered_masks)
plt.axis('off')
plt.show()

csv_filename = "colony_measurements.csv"

# Prepare the header and data list
header = ['Colony_ID', 'Center_X', 'Center_Y', 'Area_Pixels']
rows = []

for i, ann in enumerate(filtered_masks):
    # Calculate the centroid (center) of the mask
    # This helps in identifying which row belongs to which colony on the image
    m_y, m_x = np.where(ann['segmentation'])
    if len(m_x) > 0:
        cx = int(np.mean(m_x))
        cy = int(np.mean(m_y))
    else:
        cx, cy = 0, 0

    # SAM provides the area in pixels automatically
    pixel_area = ann['area']
    
    rows.append([i + 1, cx, cy, pixel_area])

# Write to file
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(rows)

print(f"Successfully exported {len(rows)} colony measurements to {csv_filename}.")