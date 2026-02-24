import cv2
import numpy as np
import csv

# 1. Load the image
image = cv2.imread('images/backlit/img1.jpg')
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray", gray)
cv2.waitKey(0)

# 2. Light Blur (Crucial for adaptive thresholding to ignore pixel-level noise)
blurred = cv2.medianBlur(gray, 5)

cv2.imshow("blur", blurred)
cv2.waitKey(0)

# 3. Detect the dish to create a mask (to avoid detecting shadows/rims)
mask = np.zeros_like(gray)
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 100, 
                           param1=50, param2=30, minRadius=400, maxRadius=600)

cv2.imshow("blur", blurred)
cv2.waitKey(0)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles[:1]:
        cv2.circle(mask, (x, y), r -90, 255, -1) # Slightly smaller to avoid edge

cv2.imshow("mask", np.where(mask == 0, mask, blurred))
cv2.waitKey(0)

# print(circles)

# 4. Adaptive Thresholding 
# This looks at a 21x21 pixel neighborhood to determine the threshold
thresh = cv2.adaptiveThreshold(blurred, 255, 
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 151 , 5)

thresh = cv2.dilate(thresh, np.ones((10,10)))

cv2.imshow("threshed", np.where(mask == 0, mask, thresh))
cv2.waitKey(0)

# better_mask = cv2.erode(mask, np.ones((80,80)), borderValue=0)

contours, hierarchy = cv2.findContours(np.where(mask == 0, mask, thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



cv2.drawContours(output, contours, -1, (0, 255, 0), 2)


cv2.imshow("output", output)
cv2.waitKey(0)
print(len(contours))
csv_data = []

minArea = 30

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if (area < minArea) :
        continue
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
        
    # Draw on the output image for visualization
    cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
    cv2.putText(output, str(i+1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Add to CSV list
    csv_data.append([i + 1, cX, cY, area])

# 7. Save to CSV
csv_filename = 'colony_data_opencv.csv'
header = ['Colony_ID', 'Center_X', 'Center_Y', 'Area_Pixels']

with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(csv_data)

print(f"Detected {len(csv_data)} colonies.")
print(f"Data saved to {csv_filename}")

# 8. Show Results
cv2.imshow("Detection with IDs", output)
cv2.waitKey(0)
cv2.destroyAllWindows()


# print(np.count_nonzero(thresh * mask))
# cv2.imshow("Adaptive Detection", better_mask )
# cv2.imshow("Adaptive Detection", better_mask * thresh * 255)
cv2.imshow("Adaptive Detection", output)
cv2.waitKey(0)