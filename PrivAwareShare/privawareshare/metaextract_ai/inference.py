
from typing import Dict, Any
import numpy as np

from .activity_model import predict_activity_demo
from .emotion_model import predict_emotion_demo
from .gesture_model import predict_gesture_demo
from .location_exif import extract_location_demo, map_location_to_context

def run_metaextract_demo(img_np: np.ndarray) -> Dict[str, Any]:
    activity_probs = predict_activity_demo()
    emotion_probs = predict_emotion_demo()
    gesture_probs = predict_gesture_demo()

    activity = max(activity_probs, key=activity_probs.get)
    emotion = max(emotion_probs, key=emotion_probs.get)
    gesture = max(gesture_probs, key=gesture_probs.get)

    lat, lon = extract_location_demo()
    loc_context = map_location_to_context(lat, lon)
    location_match = 1.0 if loc_context == "home" else 0.0

    metadata_consistency = 0.9  # high, demo

    return {
        "activity": activity,
        "activity_prob": activity_probs[activity],
        "emotion": emotion,
        "emotion_prob": emotion_probs[emotion],
        "gesture": gesture,
        "gesture_flag": 0.0 if gesture == "none" else 1.0,
        "location": loc_context,
        "location_match": location_match,
        "metadata_consistency": metadata_consistency,
    }
