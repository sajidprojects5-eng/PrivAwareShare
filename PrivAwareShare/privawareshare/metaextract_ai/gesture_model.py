
import random
from typing import Dict, List

GESTURE_CLASSES: List[str] = ["none", "thumbs_up", "wave", "pointing"]

def predict_gesture_demo() -> Dict[str, float]:
    scores = [random.random() for _ in GESTURE_CLASSES]
    s = sum(scores)
    probs = [x / s for x in scores]
    return {c: p for c, p in zip(GESTURE_CLASSES, probs)}
