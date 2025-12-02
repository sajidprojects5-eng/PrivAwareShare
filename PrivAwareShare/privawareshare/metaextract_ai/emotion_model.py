
import random
from typing import Dict, List

EMOTION_CLASSES: List[str] = ["happy", "sad", "angry", "neutral"]

def predict_emotion_demo() -> Dict[str, float]:
    scores = [random.random() for _ in EMOTION_CLASSES]
    s = sum(scores)
    probs = [x / s for x in scores]
    return {c: p for c, p in zip(EMOTION_CLASSES, probs)}
