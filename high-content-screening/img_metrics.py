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

def pixel_histogram(image_path, bins=10):
    """
    Calculates the pixel value histogram for an image using variable buckets.

    Args:
        image_path (str): The path to the input image.
        bins (int or list): If an int, defines the number of equal-width buckets 
                            (e.g., 10 buckets). If a list (e.g., [0, 85, 170, 256]), 
                            it defines the exact edges of custom-width buckets.
    
    Returns:
        tuple: (hist, bin_edges)
            - hist (numpy.ndarray): The pixel count inside each bucket.
            - bin_edges (numpy.ndarray): The boundary values for each bucket.
    """
    # 1. Load the image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Error: Could not load the image at {image_path}. Please check the path.")

    # 2. Convert to grayscale 
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # 3. Calculate Histogram using np.histogram
    # .ravel() flattens the 2D image matrix into a 1D list of pixels
    hist, bin_edges = np.histogram(gray_img.ravel(), bins=bins, range=(0, 256))
        
    return hist, bin_edges