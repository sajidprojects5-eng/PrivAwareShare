
import argparse
import json
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

from .preprocessing import preprocess_image
from ..privnet_cnn.inference import extract_face_embedding, dummy_face_database
from ..metaextract_ai.inference import run_metaextract_demo
from ..policymatch_net.inference import compute_privacy_score
from ..sharedecision_net.inference import final_sharing_decision_demo
from ..privanon_gan.anonymize import anonymize_image_demo

@dataclass
class DecisionResult:
    decision: str
    privacy_score: float
    details: Dict[str, Any]

def run_pipeline(image_path: str, policies_path: str) -> DecisionResult:
    # 1. Preprocess
    img = preprocess_image(image_path)

    # 2. Face embedding
    emb = extract_face_embedding(img)
    db_embeddings, db_labels = dummy_face_database()
    # demo: compare with first co-publisher
    sim_num = float(np.dot(emb, db_embeddings[0]))
    sim_den = float(np.linalg.norm(emb) * np.linalg.norm(db_embeddings[0]) + 1e-8)
    face_similarity = sim_num / sim_den

    # 3. Metadata
    metadata = run_metaextract_demo(img)

    # 4. Policies
    with open(policies_path, "r") as f:
        policies = json.load(f)
    # demo: assume first co-publisher policy
    policy = policies.get(db_labels[0], {})

    # 5. PolicyMatchNet score
    privacy_score = compute_privacy_score(
        face_similarity=face_similarity,
        activity_prob=metadata["activity_prob"],
        emotion_prob=metadata["emotion_prob"],
        location_match=metadata["location_match"],
        gesture_flag=metadata["gesture_flag"],
    )

    # 6. Final decision
    decision, detail = final_sharing_decision_demo(
        face_similarity=face_similarity,
        metadata_consistency=metadata["metadata_consistency"],
        policy_score=privacy_score,
    )

    # 7. Optional anonymization preview
    if decision == "ANONYMIZE":
        _ = anonymize_image_demo(img)  # not saved in this demo

    return DecisionResult(
        decision=decision,
        privacy_score=privacy_score,
        details={
            "face_similarity": face_similarity,
            "metadata": metadata,
            "policy": policy,
            "decision_detail": detail,
        },
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--policies_path", required=True)
    args = parser.parse_args()

    result = run_pipeline(args.image_path, args.policies_path)
    print("=== PrivAwareShare Demo Result ===")
    print("Decision:", result.decision)
    print("Privacy score:", f"{result.privacy_score:.3f}")
    print("Details:", json.dumps(result.details, indent=2))

if __name__ == "__main__":
    main()
