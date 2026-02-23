import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt

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
image = cv2.imread('images/backlit/img4.jpg')

# Resize if the image is massive (e.g., > 2000px wide) to speed up inference
# SAM works best around 1024-1500px
height, width = image.shape[:2]
max_dim = 1000
if max(height, width) > max_dim:
    scale = max_dim / max(height, width)
    image = cv2.resize(image, None, fx=scale, fy=scale)
    print(f"Resized image to {image.shape[:2]} for speed.")



# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
# This makes faint colonies dark and distinct, so SAM detects them easier
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(image_lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
limg = cv2.merge((cl,a,b))
enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

print("Generating masks (this might still take 30-60s on CPU)...")
masks = mask_generator.generate(enhanced_image)

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
plt.imshow(enhanced_image)
show_anns(filtered_masks)
plt.axis('off')
plt.show()