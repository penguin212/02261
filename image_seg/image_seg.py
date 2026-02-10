import cv2
import numpy as np

# 1. Load the image
image = cv2.imread('images/backlit/img4.jpg')
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Light Blur (Crucial for adaptive thresholding to ignore pixel-level noise)
blurred = cv2.medianBlur(gray, 5)

# 3. Detect the dish to create a mask (to avoid detecting shadows/rims)
mask = np.zeros_like(gray)
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 100, 
                           param1=50, param2=30, minRadius=400, maxRadius=600)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles[:1]:
        cv2.circle(mask, (x, y), r -90, 255, -1) # Slightly smaller to avoid edge

# 4. Adaptive Thresholding 
# This looks at a 21x21 pixel neighborhood to determine the threshold
thresh = cv2.adaptiveThreshold(blurred, 255, 
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 151 , 5)

thresh = cv2.dilate(thresh, np.ones((10,10)))

# better_mask = cv2.erode(mask, np.ones((80,80)), borderValue=0)

contours, hierarchy = cv2.findContours(thresh * mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

print(len(contours))



print(np.count_nonzero(thresh * mask))
# cv2.imshow("Adaptive Detection", better_mask )
# cv2.imshow("Adaptive Detection", better_mask * thresh * 255)
cv2.imshow("Adaptive Detection", output)
cv2.waitKey(0)