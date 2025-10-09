import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from paths import FIGURES


def savefig(name: str):
    """Save current Matplotlib figure into FIGURES/"""
    plt.savefig(FIGURES / name, dpi=150, bbox_inches="tight")


def evaluate_svm(model, X, y, cv, label, color):
    tprs, aucs, accuracies = [], [], []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6, 5))

    for i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)

        # --- Accuracy ---
        # (TP+TN) / (TP+TN+FP+FN)
        y_pred = model.predict(X_val)
        acc = (y_pred == y_val).mean()
        accuracies.append(acc)

        # --- ROC & AUC ---
        y_prob = model.decision_function(X_val)
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        ax.plot(fpr, tpr, lw=1, alpha=0.5, label=f"Fold {i+1} (AUC = {roc_auc:.2f})")

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc, std_auc = auc(mean_fpr, mean_tpr), np.std(aucs)
    mean_acc, std_acc = np.mean(accuracies), np.std(accuracies)

    ax.plot(mean_fpr, mean_tpr, color=color,
            label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})", lw=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title=f"ROC Curve - {label}")
    ax.legend()
    plt.tight_layout()

    return fig, mean_auc, std_auc, mean_acc, std_acc

