"""Metrics, confidence intervals and plots used by train_model.py.

All figures are saved as PNG files into outputs/ so they can be inserted into
the report and presentation. Nothing here contains hard-coded results.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)

from src.config import NUMERIC_FEATURES, TARGET

CLASS_NAMES = ["Non-liver (0)", "Liver (1)"]
sns.set_theme(style="whitegrid", context="notebook")


# --------------------------------------------------------------------------- metrics
def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Metrics with class 1 (liver-patient class) as the positive class."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if y_prob is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
        out["brier_score"] = brier_score_loss(y_true, y_prob)
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def bootstrap_ci(y_true, y_pred, y_prob, n_boot=1000, seed=42, alpha=0.05) -> dict:
    """Percentile bootstrap confidence intervals on the held-out test set.

    The test set is small, so a point estimate alone can be misleading.
    """
    rng = np.random.default_rng(seed)
    y_true, y_pred, y_prob = map(np.asarray, (y_true, y_pred, y_prob))
    samples = {"roc_auc": [], "recall_sensitivity": [], "precision": [], "f1": []}
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        samples["roc_auc"].append(roc_auc_score(y_true[idx], y_prob[idx]))
        samples["recall_sensitivity"].append(recall_score(y_true[idx], y_pred[idx], zero_division=0))
        samples["precision"].append(precision_score(y_true[idx], y_pred[idx], zero_division=0))
        samples["f1"].append(f1_score(y_true[idx], y_pred[idx], zero_division=0))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))]
            for k, v in samples.items() if v}


# --------------------------------------------------------------------------- EDA plots
def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(df: pd.DataFrame, path):
    counts = df[TARGET].value_counts().reindex([1, 0]).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(["Liver patient (1)", "Non-liver patient (0)"], counts.values,
                  color=["#1f4e79", "#2a9d8f"])
    total = counts.sum()
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({100 * v / total:.1f}%)",
                ha="center", va="bottom")
    ax.set_ylabel("Number of records")
    ax.set_title("Class distribution")
    ax.set_ylim(0, counts.max() * 1.2)
    _save(fig, path)


def plot_feature_distributions(df: pd.DataFrame, path):
    cols = NUMERIC_FEATURES
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for ax, col in zip(axes.ravel(), cols):
        sns.boxplot(data=df, x=TARGET, y=col, hue=TARGET, palette=["#2a9d8f", "#1f4e79"],
                    legend=False, ax=ax, showfliers=True, fliersize=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Non-liver (0)", "Liver (1)"])
        ax.set_xlabel("")
        ax.set_title(col.replace("_", " "))
        if df[col].max() > 20 * max(df[col].median(), 1e-9):
            ax.set_yscale("log")  # heavy right skew in enzymes / bilirubin
    fig.suptitle("Feature distributions by class (log scale where heavily skewed)", y=1.0)
    _save(fig, path)


def plot_correlation(df: pd.DataFrame, path):
    data = df[NUMERIC_FEATURES + [TARGET]].copy()
    corr = data.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Spearman correlation matrix")
    _save(fig, path)


def plot_missing_values(df: pd.DataFrame, path):
    miss = df.drop(columns=[TARGET]).isna().sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    miss.plot(kind="barh", color="#1f4e79", ax=ax)
    ax.set_xlabel("Missing values")
    ax.set_title("Missing values per feature")
    _save(fig, path)


# --------------------------------------------------------------------------- model plots
def plot_confusion_matrix(y_true, y_pred, path, title="Confusion matrix (test set)"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, annot_kws={"size": 14})
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(title)
    _save(fig, path)


def plot_confusion_matrices_grid(results: dict, path):
    """results: {model_name: (y_true, y_pred)}"""
    n = len(results)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.2 * rows))
    for ax, (name, (yt, yp)) in zip(np.atleast_1d(axes).ravel(), results.items()):
        sns.heatmap(confusion_matrix(yt, yp, labels=[0, 1]), annot=True, fmt="d", cmap="Blues",
                    cbar=False, ax=ax, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    _save(fig, path)


def plot_roc_curves(roc_data: dict, path):
    """roc_data: {model_name: (y_true, y_prob)}"""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, (yt, yp) in roc_data.items():
        fpr, tpr, _ = roc_curve(yt, yp)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc_score(yt, yp):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def plot_model_comparison(table: pd.DataFrame, path):
    metrics = ["cv_roc_auc", "test_roc_auc", "test_recall_sensitivity", "test_precision", "test_f1"]
    long = table.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=long, x="model", y="score", hue="metric", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_title("Model comparison")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, path)


def plot_calibration(y_true, prob_dict: dict, path):
    """prob_dict: {label: probabilities}; closer to the diagonal = better calibrated."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    for name, probs in prob_dict.items():
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=5, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction of liver-patient class")
    ax.set_title("Reliability (calibration) curve")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_permutation_importance(model, X, y, path, n_repeats=20, seed=42) -> pd.DataFrame:
    """Permutation importance (drop in ROC-AUC when a feature is shuffled) on the test set."""
    res = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=n_repeats,
                                 random_state=seed, n_jobs=1)
    imp = pd.DataFrame({"feature": X.columns, "importance_mean": res.importances_mean,
                        "importance_std": res.importances_std}).sort_values(
        "importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(imp["feature"], imp["importance_mean"], xerr=imp["importance_std"], color="#1f4e79")
    ax.set_xlabel("Mean decrease in ROC-AUC")
    ax.set_title("Permutation feature importance (test set)")
    _save(fig, path)
    return imp.sort_values("importance_mean", ascending=False)
