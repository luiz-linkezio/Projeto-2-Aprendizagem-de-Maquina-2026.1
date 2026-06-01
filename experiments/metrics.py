from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    n_classes = len(np.unique(y_true))

    if n_classes == 2:
        # Mambular pode retornar (n,1) para binário — usa coluna 0 nesse caso
        proba_col = y_proba[:, 0] if y_proba.shape[1] == 1 else y_proba[:, 1]
        auc = float(roc_auc_score(y_true, proba_col))
    else:
        auc = float(roc_auc_score(y_true, y_proba, multi_class="ovo", average="macro"))

    acc = float(accuracy_score(y_true, y_pred))

    # G-Mean: média geométrica da recall por classe
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        mask = y_true == c
        recalls.append(float((y_pred[mask] == c).sum() / mask.sum()) if mask.sum() > 0 else 0.0)
    g_mean = float(np.prod(recalls) ** (1.0 / len(recalls))) if all(r > 0 for r in recalls) else 0.0

    ce = float(log_loss(y_true, y_proba))

    return {"auc_ovo": auc, "acc": acc, "g_mean": g_mean, "ce": ce}
