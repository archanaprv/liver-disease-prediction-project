"""Dataset loading, data-quality inspection and the scikit-learn preprocessing pipeline.

Leakage control: every statistic (imputation medians, scaling mean/std, encoder
categories) is learned inside a scikit-learn Pipeline, so it is fitted ONLY on
training data (or on the training folds during cross-validation).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES, COLUMN_ALIASES, DATA_PATH, FEATURES, FEATURE_SPECS, GENDER_VALUES,
    LOG_FEATURES, NUMERIC_FEATURES, OTHER_NUMERIC_FEATURES, TARGET,
    TARGET_ALIASES, UCI_COLUMN_ORDER,
)

DATASET_HELP = """
Dataset file not found: {path}

How to fix:
  1. Download the Indian Liver Patient Dataset (ILPD), either
       - UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset
       - or the Kaggle copy 'indian_liver_patient.csv'
  2. Save the CSV as: {default}
     (or pass a different location with  --data <path>)
  3. Re-run:  python train_model.py
"""


def _normalise(name: str) -> str:
    """'A/G Ratio' -> 'agratio'; used to match differently named columns."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load ILPD (UCI headerless file or Kaggle CSV) into a standard schema.

    Returns a DataFrame with the canonical FEATURES columns plus TARGET, where
    TARGET = 1 for the dataset's 'liver patient' class (original label 1) and
    0 for the 'non-liver patient' class (original label 2).
    Invalid values (non-numeric, <= 0) are converted to NaN and counted in
    ``df.attrs['invalid_values_set_to_nan']``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(DATASET_HELP.format(path=path, default=DATA_PATH))

    with open(path, encoding="utf-8-sig") as fh:
        first_cell = fh.readline().strip().split(",")[0]
    if _is_number(first_cell):  # UCI file has no header row
        raw = pd.read_csv(path, header=None, names=UCI_COLUMN_ORDER)
    else:
        raw = pd.read_csv(path)

    # Rename columns to canonical names
    rename, target_col = {}, None
    for col in raw.columns:
        key = _normalise(col)
        if key in COLUMN_ALIASES:
            rename[col] = COLUMN_ALIASES[key]
        elif key in TARGET_ALIASES and target_col is None:
            target_col = col
    raw = raw.rename(columns=rename)
    missing_cols = [c for c in FEATURES if c not in raw.columns]
    if missing_cols or target_col is None:
        raise ValueError(
            f"Dataset columns do not match ILPD. Missing features: {missing_cols}; "
            f"target column found: {target_col is not None}. Columns present: {list(raw.columns)}"
        )

    # Target: original ILPD coding is 1 = liver patient, 2 = non-liver patient
    target_raw = pd.to_numeric(raw[target_col], errors="coerce")
    bad_target = ~target_raw.isin([1, 2])
    if bad_target.all():
        raise ValueError("Target column does not contain the expected labels 1 and 2.")

    df = raw[FEATURES].copy()
    df[TARGET] = (target_raw == 1).astype(int)
    df = df.loc[~bad_target].copy()

    # Clean gender text; unknown spellings become missing
    gender = df["Gender"].astype(str).str.strip().str.title()
    df["Gender"] = gender.where(gender.isin(GENDER_VALUES), np.nan)

    # Numeric coercion; permit zero only where the input specification allows it.
    invalid = 0
    for col in NUMERIC_FEATURES:
        original_missing = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        invalid_values = (
            df[col] < 0 if FEATURE_SPECS[col]["min"] == 0 else df[col] <= 0
        )
        invalid += int(original_missing.sum() + invalid_values.sum())
        df.loc[invalid_values, col] = np.nan
    df.attrs["invalid_values_set_to_nan"] = invalid
    df.attrs["rows_with_unknown_target_dropped"] = int(bad_target.sum())
    return df.reset_index(drop=True)


def data_quality_report(df: pd.DataFrame) -> dict:
    """Summarise size, missing values, duplicates, class balance (all computed, none assumed)."""
    counts = df[TARGET].value_counts().to_dict()
    pos, neg = int(counts.get(1, 0)), int(counts.get(0, 0))
    return {
        "n_records": int(len(df)),
        "n_features": len(FEATURES),
        "missing_values_per_column": {c: int(v) for c, v in df[FEATURES].isna().sum().items()},
        "total_missing_values": int(df[FEATURES].isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "invalid_values_set_to_nan": int(df.attrs.get("invalid_values_set_to_nan", 0)),
        "class_counts": {"liver_patient_1": pos, "non_liver_patient_0": neg},
        "class_percentages": {
            "liver_patient_1": round(100 * pos / max(len(df), 1), 2),
            "non_liver_patient_0": round(100 * neg / max(len(df), 1), 2),
        },
        "imbalance_ratio_majority_to_minority": round(max(pos, neg) / max(min(pos, neg), 1), 3),
        "gender_counts": {str(k): int(v) for k, v in df["Gender"].value_counts(dropna=False).items()},
        "numeric_summary": df[NUMERIC_FEATURES].describe().round(3).to_dict(),
    }


def build_preprocessor() -> ColumnTransformer:
    """Create the (unfitted) preprocessing transformer.

    * Skewed lab values : median impute -> log1p -> standardise
    * Other numeric     : median impute -> standardise
    * Gender            : most-frequent impute -> one-hot (binary column)
    """
    log_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("log_numeric", log_pipe, LOG_FEATURES),
            ("numeric", num_pipe, OTHER_NUMERIC_FEATURES),
            ("categorical", cat_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_pipeline(classifier) -> Pipeline:
    """Preprocessor + classifier as ONE estimator (guarantees identical train/serve logic)."""
    return Pipeline([("prep", build_preprocessor()), ("clf", classifier)])
