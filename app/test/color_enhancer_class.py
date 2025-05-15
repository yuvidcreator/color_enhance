import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



class ColorEnhancer:
    def __init__(self, target_rgb=(251, 213, 255)):
        self.target_rgb = np.array(target_rgb)
        self.target_hsv = cv2.cvtColor(np.uint8([[self.target_rgb]]), cv2.COLOR_RGB2HSV)[0][0]

    def _get_hsv_scaling_factor(self, detected_rgb):
        detected_hsv = cv2.cvtColor(np.uint8([[detected_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
        scaling_factor = self.target_hsv[2] / detected_hsv[2] if detected_hsv[2] > 0 else 1.0
        return scaling_factor

    def _apply_brightness_adjustment(self, image_rgb, scaling_factor):
        hsv_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2] * scaling_factor, 0, 255)
        return cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def enhance_by_kmeans(self, image_rgb, n_clusters=3):
        img_reshaped = image_rgb.reshape((-1, 3))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(img_reshaped)
        centers = kmeans.cluster_centers_.astype(int)

        distances = np.linalg.norm(centers - self.target_rgb, axis=1)
        closest_center = centers[np.argmin(distances)]
        dominant_label = np.argmin(distances)
        mask = (labels == dominant_label).astype(np.uint8).reshape(image_rgb.shape[:2]) * 255

        masked_pixels = image_rgb[mask > 0]
        mean_rgb_detected = np.mean(masked_pixels, axis=0).astype(int) if masked_pixels.size else np.array([0, 0, 0])

        scaling_factor = self._get_hsv_scaling_factor(mean_rgb_detected)
        brightened_img = self._apply_brightness_adjustment(image_rgb, scaling_factor)

        return brightened_img, mask, mean_rgb_detected, scaling_factor

    def enhance_by_hsv(self, image_rgb, detected_rgb, hsv_tolerance=(10, 60, 60)):
        hsv_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        detected_hsv = cv2.cvtColor(np.uint8([[detected_rgb]]), cv2.COLOR_RGB2HSV)[0][0]

        lower_hsv = np.clip(detected_hsv - hsv_tolerance, 0, 255)
        upper_hsv = np.clip(detected_hsv + hsv_tolerance, 0, 255)
        mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

        masked_pixels = image_rgb[mask > 0]
        mean_rgb_detected = np.mean(masked_pixels, axis=0).astype(int) if masked_pixels.size else np.array([0, 0, 0])

        scaling_factor = self._get_hsv_scaling_factor(mean_rgb_detected)
        brightened_img = self._apply_brightness_adjustment(image_rgb, scaling_factor)

        return brightened_img, mask, mean_rgb_detected, scaling_factor

    def enhance_by_hybrid(self, image_rgb, n_clusters=3, hsv_tolerance=(10, 60, 60)):
        _, mask_kmeans, center_rgb, _ = self.enhance_by_kmeans(image_rgb, n_clusters)
        hsv_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        detected_hsv_center = cv2.cvtColor(np.uint8([[center_rgb]]), cv2.COLOR_RGB2HSV)[0][0]

        lower_hsv = np.clip(detected_hsv_center - hsv_tolerance, 0, 255)
        upper_hsv = np.clip(detected_hsv_center + hsv_tolerance, 0, 255)
        mask_hsv = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

        combined_mask = cv2.bitwise_and(mask_kmeans, mask_hsv)

        masked_pixels = image_rgb[combined_mask > 0]
        mean_rgb_detected = np.mean(masked_pixels, axis=0).astype(int) if masked_pixels.size else np.array([0, 0, 0])

        scaling_factor = self._get_hsv_scaling_factor(mean_rgb_detected)
        brightened_img = self._apply_brightness_adjustment(image_rgb, scaling_factor)

        return brightened_img, combined_mask, mean_rgb_detected, scaling_factor

    def plot_results(self, image_rgb, mask, brightened_img, title='Enhanced Image', mean_rgb_before=None, mean_rgb_after=None):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image_rgb)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Detected Mask')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(brightened_img)
        title_text = title
        if mean_rgb_before is not None and mean_rgb_after is not None:
            title_text += f"\nBefore RGB: {mean_rgb_before}, After RGB: {mean_rgb_after}"
        plt.title(title_text)
        plt.axis('off')
        plt.show()



"""
# Read image
image = cv2.imread('/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/Color_Detection/Experiment_1/input_data/P237C_35.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize Enhancer
enhancer = ColorEnhancer(target_rgb=(251, 213, 255))

# Hybrid Method
bright_img_hybrid, mask_hybrid, mean_before, scaling_factor = enhancer.enhance_by_hybrid(image_rgb)

# Plot
enhancer.plot_results(image_rgb, mask_hybrid, bright_img_hybrid,
                        title='Hybrid Enhancement',
                        mean_rgb_before=mean_before,
                        mean_rgb_after=np.mean(bright_img_hybrid[mask_hybrid > 0], axis=0).astype(int)
                    )
"""


"""
# Advantages of this Class:
    - Clean and modular.
    - Switch between enhance_by_kmeans(), enhance_by_hsv(), and enhance_by_hybrid().
    - Easily reusable in any project or pipeline.
    - Supports plotting, analysis, and scaling calculation.
"""