import cv2
import numpy as np
import csv

# 1. Load the image
image = cv2.imread('groundtruth/img2.png')
# Convert BGR to RGB for easier color logic
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. Create a mask for the red areas
# Since you mentioned the only colors are red and white, 
# we look for anything that isn't white.
lower_red = np.array([100, 0, 0])   # Adjust if red is very faint
upper_red = np.array([255, 100, 100])
mask = cv2.inRange(image_rgb, lower_red, upper_red)

# 3. Find connected components (the shapes)
# cv2.RETR_EXTERNAL ensures we don't count holes inside shapes
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 4. Extract data and save to CSV
csv_filename = "red_shape_areas.csv"
data_summary = []

# Create an output image to visualize the count
output_img = image.copy()

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    
    # Calculate center for labeling
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # Filter out tiny single-pixel noise if necessary
    if area > 0:
        data_summary.append([i + 1, cX, cY, area])
        # Label the image
        cv2.putText(output_img, str(i + 1), (cX, cY), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# 5. Write to CSV
header = ['Shape_ID', 'Center_X', 'Center_Y', 'Area_Pixels']
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data_summary)

print(f"Detected {len(data_summary)} red shapes.")
print(f"Results saved to {csv_filename}")

# 6. Display results
cv2.imshow("Counted Shapes", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()