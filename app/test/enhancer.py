import cv2
import numpy as np
import json
from sklearn.cluster import KMeans
from colorsys import rgb_to_hsv, hsv_to_rgb
from scipy.spatial import distance



"""
Enforces fixed circle sequence from JSON.

Detects exactly 18 circles (left column first, right column next) based on x-coordinate split.

Calculates V scaling factor per circle using KMeans extracted dominant color.

Applies the average global V enhancement factor to the whole image, ensuring natural look.

Avoids over-enhancement using safe clipping (0.5 to 2.0 scaling factors).

The output is robust for different lighting conditions, cameras, or minor exposure issues.
"""

class ColorEnhancer:
    @staticmethod
    def calculate_v_scaling(target_rgb, current_rgb):
        # Convert to HSV (values between 0 and 1)
        target_hsv = rgb_to_hsv(*[c / 255.0 for c in target_rgb])
        current_hsv = rgb_to_hsv(*[c / 255.0 for c in current_rgb])
        v_ratio = target_hsv[2] / (current_hsv[2] + 1e-5)
        return v_ratio

    @staticmethod
    def apply_v_scaling(image, scaling_factor):
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype('float32')
        hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2] * scaling_factor, 0, 255)
        return cv2.cvtColor(hsv_img.astype('uint8'), cv2.COLOR_HSV2BGR)




class CircleColorGlobalCalibrator:
    def __init__(self, ref_json_path):
        with open(ref_json_path, 'r') as file:
            self.ref_data = json.load(file)
        self.ref_sequence = self.get_ref_sequence()

    def get_ref_sequence(self):
        sequence = []
        # Left column pink tones (top to bottom)
        for shade in ["35", "55", "70", "85", "100"]:
            sequence.append(self.ref_data["P237C"][shade])
        for shade in ["70", "80", "90", "100"]:
            sequence.append(self.ref_data["P218C"][shade])

        # Right column blue tones (top to bottom)
        for shade in ["35", "50", "65"]:
            sequence.append(self.ref_data["P310C"][shade])
        for shade in ["65", "85", "100"]:
            sequence.append(self.ref_data["P298C"][shade])
        for shade in ["65", "80", "100"]:
            sequence.append(self.ref_data["P301C"][shade])
        return sequence  # List of dict with 'rgb' keys

    def detect_circles(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
            param1=50, param2=30, minRadius=15, maxRadius=35
        )
        if circles is not None:
            circles = sorted(circles[0, :], key=lambda c: (c[0] > image.shape[1]//2, c[1]))
            return circles
        return []

    def extract_dominant_color_kmeans(self, image, x, y, r, n_clusters=3):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.circle(mask, (int(x), int(y)), int(r*0.8), 255, -1)
        masked_img = cv2.bitwise_and(image, image, mask=mask)
        pixels = masked_img[mask > 0].reshape(-1, 3)

        if len(pixels) < n_clusters:
            return np.mean(pixels, axis=0).astype(int).tolist()

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        kmeans.fit(pixels)
        dominant = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
        return dominant.astype(int).tolist()

    def process_image(self, image_path, output_path):
        image = cv2.imread(image_path)
        result_image = image.copy()
        circles = self.detect_circles(image)

        if len(circles) != 18:
            raise Exception(f"Expected 18 circles, found {len(circles)}")

        scaling_factors = []
        for i, (x, y, r) in enumerate(circles):
            current_rgb = self.extract_dominant_color_kmeans(image, x, y, r)
            target_rgb = self.ref_sequence[i]["rgb"]
            v_scaling = ColorEnhancer.calculate_v_scaling(target_rgb, current_rgb)
            scaling_factors.append(v_scaling)
            print(f"Circle {i+1}: Detected RGB {current_rgb}, Target RGB {target_rgb}, V Scaling {v_scaling:.3f}")

        # Average scaling (remove outliers by using percentiles if necessary)
        scaling_factors = np.clip(scaling_factors, 0.5, 2.0)  # Avoid extreme values
        avg_scaling = np.mean(scaling_factors)
        print(f"\nGlobal V Scaling Factor to apply: {avg_scaling:.3f}")

        # Apply global enhancement
        result_image = ColorEnhancer.apply_v_scaling(result_image, avg_scaling)
        cv2.imwrite(output_path, result_image)
        print(f"Saved globally enhanced image at: {output_path}")


# Usage
calibrator = CircleColorGlobalCalibrator('reference_data/ref_colors_data.json')
calibrator.process_image('reference_data/refernce_image.jpg', 'output_data/enhanced_image_by_enhancer_class.jpeg')
