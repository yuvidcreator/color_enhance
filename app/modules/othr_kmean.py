import cv2
import numpy as np
import json
from sklearn.cluster import KMeans
from colorsys import rgb_to_hsv, hsv_to_rgb
from scipy.spatial import distance




class CircleColorKMeansCalibrator:
    def __init__(self, ref_json_path):
        with open(ref_json_path, 'r') as file:
            self.ref_data = json.load(file)
        self.ref_colors = self.extract_ref_colors()

    def extract_ref_colors(self):
        ref_colors = []
        for key, shades in self.ref_data.items():
            for shade, color_info in shades.items():
                ref_colors.append({
                    "name": f"{key}_{shade}",
                    "rgb": color_info["rgb"],
                    "hsv": color_info["hsv"]
                })
        return ref_colors

    def detect_circles(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=50, param2=30, minRadius=10, maxRadius=30
        )
        return circles[0, :] if circles is not None else []

    def extract_dominant_color_kmeans(self, image, x, y, r, n_clusters=3):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.circle(mask, (int(x), int(y)), int(r*0.8), 255, -1) # type: ignore
        masked_img = cv2.bitwise_and(image, image, mask=mask)
        pixels = masked_img[mask > 0].reshape(-1, 3)

        if len(pixels) < n_clusters:
            return np.mean(pixels, axis=0).astype(int).tolist() # type: ignore

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        kmeans.fit(pixels)
        dominant = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
        return dominant.astype(int).tolist()

    def compare_with_ref(self, color_rgb):
        min_dist = float('inf')
        closest_ref = None
        for ref in self.ref_colors:
            dist = distance.euclidean(color_rgb, ref["rgb"])
            if dist < min_dist:
                min_dist = dist
                closest_ref = ref
        return closest_ref, min_dist

    def enhance_image_to_match_color(self, image, target_rgb, current_rgb):
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype('float32')

        # Convert to HSV to compute adjustment for the V channel (brightness)
        target_v = rgb_to_hsv(*[c / 255.0 for c in target_rgb])[2]
        current_v = rgb_to_hsv(*[c / 255.0 for c in current_rgb])[2]
        v_ratio = target_v / (current_v + 1e-5)

        # Apply ratio to V channel and clip
        hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2] * v_ratio, 0, 255)

        enhanced_bgr = cv2.cvtColor(hsv_img.astype('uint8'), cv2.COLOR_HSV2BGR)
        return enhanced_bgr

    def process_image(self, image_path, output_path, threshold=40):
        image = cv2.imread(image_path)
        result_image = image.copy()
        circles = self.detect_circles(image)

        for (x, y, r) in circles:
            color_rgb = self.extract_dominant_color_kmeans(image, x, y, r)
            ref_color, dist = self.compare_with_ref(color_rgb)
            print(f"Detected (KMeans): {color_rgb}, Closest Ref: {ref_color['name']} {ref_color['rgb']}, Distance: {dist}") # type: ignore

            if dist > threshold:
                print("Applying KMeans-based enhancement...")
                result_image = self.enhance_image_to_match_color(result_image, ref_color["rgb"], color_rgb) # type: ignore

        cv2.imwrite(output_path, result_image)
        print(f"Saved enhanced image at: {output_path}")

# # Usage
kmeans_calibrator = CircleColorKMeansCalibrator('reference_data/ref_colors_data.json')
kmeans_calibrator.process_image('reference_data/refernce_image.jpg', 'output_data/enhanced_by_other_kmean.jpeg')
