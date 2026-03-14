import cv2
import numpy as np

def count_255s(image_path, threshold=255):
    """
    Calculates the number of overexposed pixels in an image.

    Args:
        image_path (str): The path to the input image.
        threshold (int): The pixel intensity value (0-255) to be considered overexposed. 
                         Defaults to 250. 255 is absolutely clipped.

    Returns:
        int: The total count of overexposed pixels.
    """
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Could not load the image at {image_path}. Please check the path.")

    # 2. Convert the image to grayscale 
    # This gives us a single luminance channel to evaluate overall brightness
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Create a boolean mask of pixels that meet or exceed the threshold
    overexposed_mask = gray_img >= threshold

    # 4. Sum the True values in the mask to get the total count
    num_overexposed = np.sum(overexposed_mask)

    return num_overexposed

def pixel_histogram(image_path):
    """
    Calculates the pixel value histogram for an image.

    Args:
        image_path (str): The path to the input image.
    
    Returns:
        numpy.ndarray: A 1D array of 256 values
    """
    # 1. Load the image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Error: Could not load the image at {image_path}. Please check the path.")

    # 2. Calculate Histogram(s)
    # Convert to grayscale for a single luminance histogram
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # cv2.calcHist(images, channels, mask, histSize, ranges)
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        
    # Flatten the 2D array (256, 1) into a 1D array (256,) for easier use
    return hist.flatten()