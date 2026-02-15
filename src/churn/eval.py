import matplotlib.pyplot as plt
import pandas as pd

def summarize_results(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        m = r.get("metrics", {}) or {}
        rows.append({
            "model": name,
            "accuracy": m.get("accuracy"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "f1": m.get("f1"),
            "roc_auc": m.get("roc_auc"),
            "model_path": r.get("model_path"),
        })
    return pd.DataFrame(rows).sort_values(by="roc_auc", ascending=False, na_position="last")

def _cm(r: dict):
    return r.get("confusion_matrix") or r.get("cm")

def plot_confusion_matrix(r: dict):
    cm = _cm(r)
    if cm is None:
        fig = plt.figure()
        plt.text(0.1, 0.5, "No confusion matrix available")
        plt.axis("off")
        return fig

    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    return fig

def plot_roc_curve(r: dict):
    roc = r.get("roc") or {}
    fpr = roc.get("fpr")
    tpr = roc.get("tpr")

    fig = plt.figure()
    if fpr is None or tpr is None:
        auc = (r.get("metrics") or {}).get("roc_auc")
        plt.text(0.1, 0.5, f"ROC-AUC: {auc}")
        plt.axis("off")
        return fig

    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    return fig
