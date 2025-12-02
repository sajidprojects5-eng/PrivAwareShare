
import cv2
import numpy as np

def anonymize_image_demo(img_np: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur to central region to simulate anonymization."""
    h, w, _ = img_np.shape
    img = (img_np * 255).astype(np.uint8)
    x1, y1 = int(w * 0.25), int(h * 0.25)
    x2, y2 = int(w * 0.75), int(h * 0.75)
    roi = img[y1:y2, x1:x2]
    roi_blur = cv2.GaussianBlur(roi, (51, 51), 0)
    img[y1:y2, x1:x2] = roi_blur
    return img.astype(np.float32) / 255.0
