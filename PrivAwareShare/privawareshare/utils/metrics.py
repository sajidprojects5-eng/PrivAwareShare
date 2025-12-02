
from typing import Dict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_classification_metrics(y_true, y_pred) -> Dict[str, float]:
    """Return accuracy, precision, recall and F1 (macro)."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
