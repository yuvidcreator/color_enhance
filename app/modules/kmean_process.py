import cv2
import numpy as np
import json
from sklearn.cluster import KMeans
from scipy.spatial import distance
from colorsys import rgb_to_hsv, hsv_to_rgb




class KMeansCircleColorCalibrator:
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

    def get_dominant_color(self, image, x, y, r, n_clusters=2):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.circle(mask, (int(x), int(y)), int(r*0.7), 255, -1) # type: ignore
        pixels = image[mask == 255].reshape(-1, 3)

        if len(pixels) == 0:
            return None

        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(pixels)
        dominant = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
        return [int(c) for c in dominant]

    def compare_with_ref(self, color_rgb):
        min_dist = float('inf')
        closest_ref = None
        for ref in self.ref_colors:
            dist = distance.euclidean(color_rgb, ref["rgb"])
            if dist < min_dist:
                min_dist = dist
                closest_ref = ref
        return closest_ref, min_dist

    def enhance_global_hsv_v_channel(self, image, target_rgb, current_rgb):
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        target_hsv = np.array(rgb_to_hsv(*[x/255 for x in target_rgb]))
        current_hsv = np.array(rgb_to_hsv(*[x/255 for x in current_rgb]))
        ratio = target_hsv[2] / (current_hsv[2] + 1e-5)
        hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2] * ratio, 0, 255)
        corrected = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        return corrected

    def process_image(self, image_path, output_path, threshold=40):
        image = cv2.imread(image_path)
        result_image = image.copy()
        circles = self.detect_circles(image)

        for (x, y, r) in circles:
            dominant_color = self.get_dominant_color(image, x, y, r)
            if dominant_color is None:
                continue

            ref_color, dist = self.compare_with_ref(dominant_color)
            print(f"Dominant: {dominant_color}, Closest Ref: {ref_color['name']} {ref_color['rgb']}, Distance: {dist}") # type: ignore

            if dist > threshold:
                print("Applying KMeans-based enhancement...")
                result_image = self.enhance_global_hsv_v_channel(result_image, ref_color["rgb"], dominant_color) # type: ignore

        cv2.imwrite(output_path, result_image)
        print(f"Saved enhanced image at: {output_path}")

# # Usage
calibrator_kmeans = KMeansCircleColorCalibrator('reference_data/ref_colors_data.json')
calibrator_kmeans.process_image('reference_data/refernce_image.jpg', 'output_data/enhanced_by_kmeans_process.jpeg')
