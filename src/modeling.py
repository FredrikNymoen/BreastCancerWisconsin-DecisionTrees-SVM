from typing import Dict, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def kfold_scores(clf, X, y, cv: int = 5, scoring: str = "accuracy", random_state: int = 42) -> Tuple[float, float]:
    """
    Returns (mean, std) cross-val score using StratifiedKFold.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, X, y, cv=skf, scoring=scoring)
    return float(scores.mean()), float(scores.std())

def evaluate_holdout(clf, X_train, y_train, X_eval, y_eval) -> Dict[str, float]:
    """
    Fit on training data and compute metrics on a holdout set.
    """
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_eval)
    proba = getattr(clf, "predict_proba", None)
    y_prob = proba(X_eval)[:,1] if proba else None

    metrics = {
        "accuracy": accuracy_score(y_eval, y_pred),
        "precision": precision_score(y_eval, y_pred, zero_division=0),
        "recall": recall_score(y_eval, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_eval, y_prob)
    return {k: float(v) for k,v in metrics.items()}