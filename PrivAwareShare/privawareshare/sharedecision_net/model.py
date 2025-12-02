
from dataclasses import dataclass

@dataclass
class ShareDecisionConfig:
    face_similarity_weight: float = 0.5
    metadata_consistency_weight: float = 0.3
    policy_score_weight: float = 0.2
    final_auto_approve_threshold: float = 0.8
    final_review_threshold: float = 0.5

class ShareDecisionNet:
    def __init__(self, cfg: ShareDecisionConfig):
        self.cfg = cfg

    def compute_final_score(
        self,
        face_similarity: float,
        metadata_consistency: float,
        policy_score: float,
    ) -> float:
        score = (
            self.cfg.face_similarity_weight * face_similarity
            + self.cfg.metadata_consistency_weight * metadata_consistency
            + self.cfg.policy_score_weight * policy_score
        )
        return float(max(0.0, min(1.0, score)))

    def decide(self, final_score: float) -> str:
        if final_score >= self.cfg.final_auto_approve_threshold:
            return "PUBLISH"
        if final_score >= self.cfg.final_review_threshold:
            return "REQUEST_APPROVAL"
        return "ANONYMIZE"
