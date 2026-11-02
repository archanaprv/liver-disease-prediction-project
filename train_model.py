"""Train, compare, evaluate and save the liver-disease classification model.

Run:  python train_model.py
Optional: python train_model.py --data data/indian_liver_patient.csv

EVALUATION PROTOCOL (documented for the report)
1. Load ILPD, report data quality, drop exact duplicate rows (avoids the same record
   landing in both train and test).
2. Stratified 80/20 train-test split with a fixed seed. The test set is untouched until the end.
3. For each candidate model: 5-fold stratified cross-validated grid search on the TRAINING
   set only (preprocessing is inside the Pipeline, so no leakage). Selection metric: ROC-AUC.
4. Best model by cross-validated ROC-AUC (tie-break: CV F1) is wrapped in sigmoid probability
   calibration (fitted on training data only).
5. The selected calibrated model is evaluated ONCE on the held-out test set.
6. The calibrated pipeline is saved with Joblib; metrics and plots go to outputs/.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src import evaluation as ev
from src.config import (
    CV_FOLDS, DATA_PATH, DECISION_THRESHOLD, FEATURES, MODELS_DIR, NUMERIC_FEATURES,
    OUTPUTS_DIR, RANDOM_STATE, TARGET, TEST_SIZE,
)
from src.preprocessing import data_quality_report, load_dataset, make_pipeline


def candidate_models(seed: int) -> dict:
    """Name -> (pipeline, parameter grid). class_weight='balanced' handles class imbalance."""
    return {
        "Logistic Regression": (
            make_pipeline(LogisticRegression(class_weight="balanced", max_iter=2000,
                                             random_state=seed)),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "Decision Tree": (
            make_pipeline(DecisionTreeClassifier(class_weight="balanced", random_state=seed)),
            {"clf__max_depth": [3, 4, 5, 6], "clf__min_samples_leaf": [5, 10, 20]},
        ),
        "Random Forest": (
            make_pipeline(RandomForestClassifier(class_weight="balanced", random_state=seed,
                                                 n_jobs=1)),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [4, 6, 8],
             "clf__min_samples_leaf": [3, 5]},
        ),
        "SVM (RBF)": (
            make_pipeline(SVC(kernel="rbf", class_weight="balanced", random_state=seed)),
            {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.1]},
        ),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Train the liver disease prediction model.")
    p.add_argument("--data", default=str(DATA_PATH), help="Path to the ILPD CSV file.")
    p.add_argument("--models-dir", default=str(MODELS_DIR))
    p.add_argument("--outputs-dir", default=str(OUTPUTS_DIR))
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--test-size", type=float, default=TEST_SIZE)
    p.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    p.add_argument("--keep-duplicates", action="store_true",
                   help="Do not remove exact duplicate rows before splitting.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir, models_dir = Path(args.outputs_dir), Path(args.models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------ 1. data
    print("[1/7] Loading dataset ...")
    try:
        df = load_dataset(args.data)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        raise SystemExit(1)
    report = data_quality_report(df)
    n_raw = len(df)
    if not args.keep_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    report["records_after_duplicate_removal"] = int(len(df))
    report["class_counts_after_duplicate_removal"] = {
        "liver_patient_1": int((df[TARGET] == 1).sum()),
        "non_liver_patient_0": int((df[TARGET] == 0).sum()),
    }
    (out_dir / "data_summary.json").write_text(json.dumps(report, indent=2))
    print(f"      records: {n_raw}, features: {len(FEATURES)}, "
          f"missing cells: {report['total_missing_values']}, "
          f"duplicates: {report['duplicate_rows']}")
    print(f"      class counts (raw): {report['class_counts']}")
    print(f"      missing per column: "
          f"{ {k: v for k, v in report['missing_values_per_column'].items() if v} }")

    # ------------------------------------------------------------------ 2. EDA
    print("[2/7] Creating exploratory plots ...")
    ev.plot_class_distribution(df, out_dir / "class_distribution.png")
    ev.plot_feature_distributions(df, out_dir / "feature_distributions.png")
    ev.plot_correlation(df, out_dir / "correlation_heatmap.png")
    ev.plot_missing_values(df, out_dir / "missing_values.png")

    # ------------------------------------------------------------------ 3. split
    print("[3/7] Stratified train-test split ...")
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed)
    print(f"      train: {len(X_train)}  test: {len(X_test)}")

    # ------------------------------------------------------------------ 4. CV tuning
    print("[4/7] Cross-validated model comparison (training data only) ...")
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    scoring = {"roc_auc": "roc_auc", "f1": "f1", "recall": "recall", "precision": "precision"}
    rows, fitted = [], {}
    for name, (pipe, grid) in candidate_models(args.seed).items():
        t0 = time.time()
        search = GridSearchCV(pipe, grid, scoring=scoring, refit="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        i = search.best_index_
        cvr = search.cv_results_
        est = search.best_estimator_
        row = {"model": name,
               "cv_roc_auc": float(cvr["mean_test_roc_auc"][i]),
               "cv_roc_auc_std": float(cvr["std_test_roc_auc"][i]),
               "cv_f1": float(cvr["mean_test_f1"][i]),
               "cv_recall": float(cvr["mean_test_recall"][i]),
               "cv_precision": float(cvr["mean_test_precision"][i]),
               "best_params": {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v)
                               for k, v in search.best_params_.items()}}
        rows.append(row)
        fitted[name] = est
        print(f"      {name:<20} CV ROC-AUC {row['cv_roc_auc']:.3f} "
              f"(+/- {row['cv_roc_auc_std']:.3f})  [{time.time() - t0:.1f}s]")

    table = pd.DataFrame(rows)
    best_name = table.sort_values(["cv_roc_auc", "cv_f1"], ascending=False).iloc[0]["model"]
    print(f"      Selected model (highest CV ROC-AUC): {best_name}")

    # ------------------------------------------------------------------ 5. calibration
    print("[5/7] Calibrating probabilities of the selected model (training data only) ...")
    calibrated = CalibratedClassifierCV(
        estimator=clone(fitted[best_name]), method="sigmoid",
        cv=StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed))
    calibrated.fit(X_train, y_train)
    cal_prob = calibrated.predict_proba(X_test)[:, 1]
    cal_pred = (cal_prob >= DECISION_THRESHOLD).astype(int)
    final_metrics = ev.compute_metrics(y_test, cal_pred, cal_prob)
    ci = ev.bootstrap_ci(y_test, cal_pred, cal_prob, seed=args.seed)

    # ------------------------------------------------------------------ 6. outputs
    print("[6/7] Saving evaluation plots and reports ...")
    ev.plot_confusion_matrix(y_test, cal_pred, out_dir / "confusion_matrix.png",
                             f"Confusion matrix - {best_name} (calibrated)")
    ev.plot_roc_curves(
        {f"{best_name} (calibrated)": (y_test, cal_prob)},
        out_dir / "roc_curves.png")
    ev.plot_model_comparison(table, out_dir / "model_comparison.png")
    ev.plot_calibration(y_test, {f"{best_name} (calibrated)": cal_prob},
                        out_dir / "calibration_curve.png")
    importance = ev.plot_permutation_importance(calibrated, X_test, y_test,
                                                out_dir / "feature_importance.png",
                                                seed=args.seed)
    importance.to_csv(out_dir / "feature_importance.csv", index=False)

    report_txt = classification_report(
        y_test, cal_pred, target_names=["Non-liver patient (0)", "Liver patient (1)"], digits=3)
    (out_dir / "classification_report.txt").write_text(
        f"Final model: {best_name} + sigmoid calibration (test set, threshold "
        f"{DECISION_THRESHOLD})\n\n{report_txt}")
    pd.DataFrame({"y_true": y_test.values, "prob_liver_class": cal_prob,
                  "y_pred": cal_pred}).to_csv(out_dir / "test_predictions.csv", index=False)
    table.drop(columns=["best_params"]).round(4).to_csv(out_dir / "model_comparison.csv",
                                                        index=False)

    training_ranges = {c: [float(X_train[c].min()), float(X_train[c].max())]
                       for c in NUMERIC_FEATURES}
    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": {"test_size": args.test_size, "cv_folds": args.cv_folds,
                     "seed": args.seed, "selection_metric": "cross-validated ROC-AUC",
                     "duplicates_removed": not args.keep_duplicates,
                     "decision_threshold": DECISION_THRESHOLD,
                     "n_train": int(len(X_train)), "n_test": int(len(X_test))},
        "models": rows,
        "selected_model": best_name,
        "final_test_metrics_calibrated": final_metrics,
        "final_test_bootstrap_95ci": ci,
        "feature_importance": importance.round(5).to_dict(orient="records"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    # ------------------------------------------------------------------ 7. save model
    print("[7/7] Saving model with Joblib ...")
    bundle = {
        "model": calibrated,            # full pipeline: preprocessing + classifier + calibration
        "features": FEATURES,
        "threshold": DECISION_THRESHOLD,
        "metadata": {
            "model_name": best_name,
            "best_params": table.set_index("model").loc[best_name, "best_params"],
            "positive_class": "1 = liver patient class (ILPD label 1)",
            "trained_at": metrics["generated_at"],
            "sklearn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "python_version": platform.python_version(),
            "training_ranges": training_ranges,
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "test_metrics": final_metrics,
        },
    }
    joblib.dump(bundle, models_dir / "liver_model.joblib")

    print("\n==== FINAL RESULT (held-out test set) ====")
    print(f"Model: {best_name} (+ calibration)")
    for k in ("accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc"):
        print(f"  {k:<20} {final_metrics[k]:.3f}")
    print(f"  confusion (tn, fp, fn, tp): {final_metrics['tn']}, {final_metrics['fp']}, "
          f"{final_metrics['fn']}, {final_metrics['tp']}")
    print(f"\nPlots and metrics saved in: {out_dir}")
    print(f"Model saved to: {models_dir / 'liver_model.joblib'}")


if __name__ == "__main__":
    main()
