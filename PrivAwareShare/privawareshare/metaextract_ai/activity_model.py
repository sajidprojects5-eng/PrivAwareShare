
import random
from typing import Dict, List

ACTIVITY_CLASSES: List[str] = ["walking", "sitting", "dancing", "swimming", "playing"]

def predict_activity_demo() -> Dict[str, float]:
    scores = [random.random() for _ in ACTIVITY_CLASSES]
    s = sum(scores)
    probs = [x / s for x in scores]
    return {c: p for c, p in zip(ACTIVITY_CLASSES, probs)}
