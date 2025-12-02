
import cv2
import numpy as np
from typing import Tuple

def preprocess_image(img_path: str, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load an image and resize/normalize to (H,W,C) float32 in [0,1]."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    return img
