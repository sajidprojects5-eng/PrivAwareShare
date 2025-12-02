
from .model import PolicyMatchConfig, PolicyMatchNet

_cfg = PolicyMatchConfig()
_net = PolicyMatchNet(_cfg)

def compute_privacy_score(
    face_similarity: float,
    activity_prob: float,
    emotion_prob: float,
    location_match: float,
    gesture_flag: float,
) -> float:
    return _net.compute_score(
        face_similarity=face_similarity,
        activity_prob=activity_prob,
        emotion_prob=emotion_prob,
        location_match=location_match,
        gesture_flag=gesture_flag,
    )
