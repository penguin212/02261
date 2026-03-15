import cv2
import numpy as np 

MIN_AREA = 200 # Minimum area of colony contour

def avg_cell_area(img_path):
    """
    Calculates the average cell area and the number of cells detected in one function call

    Args:
        img_path (str): The path to the input image.

    Returns: (float, int)
        float: average pixel count for each cell found in image
        int: number of cells detected
    """
    # 1. Load the image
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Error: Could not load the image at {img_path}. Please check the path.")
    
    #output_all = image.copy()
    #output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # 2. Light Blur 
    blurred = cv2.medianBlur(gray, 3)

    #cv2.imshow("blur", blurred)
    #cv2.waitKey(0)

    # 3. Adaptive Thresholding 
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 851, 13)
    thresh = cv2.dilate(thresh, np.ones((10, 10), np.uint8))

    # 4. Find and draw contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #print(f"Total Contours Found: {len(contours)}")
    #cv2.drawContours(output_all, contours, -1, (0, 0, 255), 1)
    #cv2.imshow("All Contours (Unfiltered)", output_all)
    
    csv_data = []
    total_area = 0

    # 5. Process each contour
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        # Skip noise based on threshold
        if area < MIN_AREA:
            continue

        #cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
            
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
            
        colony_data = {"id": len(csv_data) + 1, "area": area, "cX": cX, "cY": cY}
        csv_data.append(colony_data)
        
        # Add to total area for the average calculation
        total_area += area

    #cv2.imshow("out", output)
    #cv2.waitKey(0)

    # 6. Output final metrics
    valid_colonies = len(csv_data)
    #print(f"Detected {valid_colonies} valid colonies (Area >= {MIN_AREA}).")
    
    if valid_colonies > 0:
        average_area = total_area / valid_colonies
    else:
        average_area = 0
        
    #print(f"Average Colony Area: {average_area} pixels.")

    return average_area, valid_colonies

"""def count_cells(img_path):

    Calculates the number of overexposed pixels in an image.

    Args:
        img_path (str): The path to the input image.

    Returns:
        int: number of cells found in the image

    # 1. Load the image
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Error: Could not load the image at {img_path}. Please check the path.")
    
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Light Blur 
    blurred = cv2.medianBlur(gray, 3)

    # 3. Adaptive Thresholding 
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 851, 13)
    thresh = cv2.dilate(thresh, np.ones((10, 10), np.uint8))

    # 4. Find and draw contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    
    total_count = 0

    # 5. Process each contour
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        # Skip noise based on threshold
        if area < MIN_AREA:
            continue
            
        total_count += 1

    #print(f"Detected {total_count} valid cells (Area >= {MIN_AREA}).")

    return total_count"""
    
#print(avg_cell_area("data/train/G3_10x_A3_F4_T0_TRANS.jpg"))
#print(count_cells("data/train/G3_10x_A3_F4_T0_TRANS.jpg"))