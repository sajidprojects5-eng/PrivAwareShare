
from dataclasses import dataclass

@dataclass
class PolicyMatchConfig:
    w_face: float = 0.4
    w_activity: float = 0.2
    w_emotion: float = 0.2
    w_location: float = 0.1
    w_gesture: float = 0.1

class PolicyMatchNet:
    """Implements a linear privacy compliance score."""

    def __init__(self, cfg: PolicyMatchConfig):
        self.cfg = cfg

    def compute_score(
        self,
        face_similarity: float,
        activity_prob: float,
        emotion_prob: float,
        location_match: float,
        gesture_flag: float,
    ) -> float:
        score = (
            self.cfg.w_face * face_similarity
            + self.cfg.w_activity * activity_prob
            + self.cfg.w_emotion * emotion_prob
            + self.cfg.w_location * location_match
            - self.cfg.w_gesture * gesture_flag
        )
        return float(max(0.0, min(1.0, score)))
