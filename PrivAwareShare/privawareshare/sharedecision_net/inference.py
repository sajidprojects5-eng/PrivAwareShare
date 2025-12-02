
from typing import Tuple, Dict, Any
from .model import ShareDecisionConfig, ShareDecisionNet

_cfg = ShareDecisionConfig()
_net = ShareDecisionNet(_cfg)

def final_sharing_decision_demo(
    face_similarity: float,
    metadata_consistency: float,
    policy_score: float,
) -> Tuple[str, Dict[str, Any]]:
    final_score = _net.compute_final_score(
        face_similarity=face_similarity,
        metadata_consistency=metadata_consistency,
        policy_score=policy_score,
    )
    decision = _net.decide(final_score)
    detail = {
        "final_score": final_score,
        "thresholds": {
            "auto_approve": _cfg.final_auto_approve_threshold,
            "review": _cfg.final_review_threshold,
        },
    }
    return decision, detail
