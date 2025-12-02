
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F

from .model import PrivNetCNN

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = PrivNetCNN(embedding_dim=512, pretrained=True).to(_device)
_model.eval()

def _to_tensor(img_np: np.ndarray) -> torch.Tensor:
    # img_np: (H,W,C) float32 [0,1]
    x = np.transpose(img_np, (2, 0, 1))
    x = torch.from_numpy(x).unsqueeze(0).to(_device)
    return x

def extract_face_embedding(img_np: np.ndarray) -> np.ndarray:
    """Treat full image as a face crop and extract embedding (demo)."""
    with torch.no_grad():
        x = _to_tensor(img_np)
        emb = _model(x)
        emb = F.normalize(emb, dim=1)
    return emb.cpu().numpy().reshape(-1)

def dummy_face_database(n_identities: int = 3, dim: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(42)
    embs = rng.randn(n_identities, dim).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    labels = np.array([f"copublisher_{i}" for i in range(n_identities)])
    return embs, labels
