import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from app.helpers.save_image import get_ouput_image_path



def detect_colored_circles(image_path):
    """Detects colored circles in the given image."""
    # Load image
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=20, 
        param1=50, 
        param2=30, 
        minRadius=5, 
        maxRadius=30
    )
    
    circle_colors = []
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]: # type: ignore
            # Get the color of the circle
            color_bgr = image[y, x]
            color_rgb = color_bgr[::-1]  # BGR to RGB
            color_hsv = cv2.cvtColor(np.uint8([[color_bgr]]), cv2.COLOR_BGR2HSV)[0][0] # type: ignore
            circle_colors.append((color_rgb.tolist(), color_hsv.tolist()))

    return circle_colors



def compare_with_reference(circle_colors, reference_colors):
    """Compares detected circle colors with reference colors and applies KMeans if needed."""
    matches = []
    for circle_color in circle_colors:
        rgb, hsv = circle_color
        match_found = False
        
        for ref_color in reference_colors:
            if np.allclose(rgb, ref_color, atol=30):  # Adjust tolerance as needed
                matches.append((rgb, hsv, ref_color, True))
                match_found = True
                break
        
        if not match_found:
            # Apply KMeans for enhancement
            cluster_colors = np.array(circle_colors)[:, 1].reshape(-1, 1, 3)
            kmeans = KMeans(n_clusters=1).fit(cluster_colors)
            enhanced_hsv = kmeans.cluster_centers_[0]
            enhanced_rgb = cv2.cvtColor(np.uint8([[enhanced_hsv]]), cv2.COLOR_HSV2BGR)[0][0] # type: ignore
            matches.append((rgb, hsv, enhanced_rgb, False))

    return matches



def enhance_image(image_path, matches):
    """Enhance the image based on color matches."""
    image = cv2.imread(image_path)
    
    for match in matches:
        original_rgb, original_hsv, target_rgb, is_match = match
        
        if not is_match:
            # Define a simple scaling on the V value for enhancement
            # Increase V channel by a specific factor (Adjust as needed)
            enhanced_hsv = np.array(original_hsv)
            enhanced_hsv[2] = min(255, enhanced_hsv[2] + 40)  # Increase V channel
            
            enhanced_bgr = cv2.cvtColor(np.uint8([[enhanced_hsv]]), cv2.COLOR_HSV2BGR)[0][0] # type: ignore
            cv2.circle(image, (original_rgb[0], original_rgb[1]), 5, enhanced_bgr.tolist(), -1)

    img_output_path, timestamp = get_ouput_image_path("output_data")
    # Save the enhanced image
    cv2.imwrite(f'{img_output_path}/{timestamp}/enhanced_image_{timestamp}.png', image)


