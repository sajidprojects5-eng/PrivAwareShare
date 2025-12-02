
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Compute cosine similarity between two 1D vectors."""
    a = a.flatten()
    b = b.flatten()
    num = float(np.dot(a, b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return num / denom
