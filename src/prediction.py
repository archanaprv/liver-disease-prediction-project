"""Input validation, model loading and prediction.

The Streamlit app is a thin layer over these functions, so the same logic can
be unit-tested without starting a web server.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    DECISION_THRESHOLD, FEATURE_SPECS, FEATURES, GENDER_VALUES, MODEL_PATH,
)

DISCLAIMER = (
    "Educational prototype only. This tool is NOT a medical device and does NOT diagnose "
    "liver disease. It reports which dataset class a set of values resembles, based on a small "
    "historical dataset. Consult a qualified healthcare professional for any medical concern."
)

MODEL_HELP = """
Trained model not found: {path}

Train it first (from the project folder):
    python train_model.py
This creates models/liver_model.joblib. Then restart the app.
"""


@dataclass
class ValidationResult:
    cleaned: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def _to_float(value):
    """Convert a user value to float; return None if blank, raise ValueError if invalid."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in {"na", "n/a", "nan", "none", "null"}:
            return None
        value = value.replace(",", ".") if value.count(",") == 1 and "." not in value else value
    number = float(value)  # raises ValueError / TypeError for junk
    if math.isnan(number):
        return None
    if math.isinf(number):
        raise ValueError("infinite value")
    return number


def validate_inputs(raw: dict, training_ranges: dict | None = None) -> ValidationResult:
    """Validate raw user input.

    * Required fields must be present, numeric and within plausibility limits.
    * Optional field (A/G ratio) may be blank -> NaN (imputed by the pipeline).
    * Cross-field checks catch impossible combinations.
    * Values outside the training-data range only produce a WARNING (extrapolation).
    """
    result = ValidationResult()

    # ---- Gender
    gender = raw.get("Gender")
    gender = str(gender).strip().title() if gender is not None else ""
    if gender not in GENDER_VALUES:
        result.errors.append(f"Gender must be one of {GENDER_VALUES}.")
    else:
        result.cleaned["Gender"] = gender

    # ---- Numeric fields
    for name, spec in FEATURE_SPECS.items():
        if name == "Gender":
            continue
        label = spec["label"]
        try:
            value = _to_float(raw.get(name))
        except (ValueError, TypeError):
            result.errors.append(f"{label}: '{raw.get(name)}' is not a valid number.")
            continue
        if value is None:
            if spec["required"]:
                result.errors.append(f"{label} is required.")
            else:
                result.cleaned[name] = np.nan
                result.warnings.append(
                    f"{label} not provided; the training-set median will be used instead.")
            continue
        if not (spec["min"] <= value <= spec["max"]):
            result.errors.append(
                f"{label}: {value:g} is outside the accepted range "
                f"{spec['min']:g} to {spec['max']:g} {spec['unit']}.")
            continue
        result.cleaned[name] = value

    # ---- Cross-field consistency
    c = result.cleaned
    if "Total_Bilirubin" in c and "Direct_Bilirubin" in c:
        if c["Direct_Bilirubin"] > c["Total_Bilirubin"]:
            result.errors.append("Direct bilirubin cannot be greater than total bilirubin.")
    if "Total_Proteins" in c and "Albumin" in c:
        if c["Albumin"] > c["Total_Proteins"]:
            result.errors.append("Albumin cannot be greater than total proteins.")

    # ---- Out-of-training-range warnings (do not block prediction)
    if training_ranges and result.is_valid:
        for name, (lo, hi) in training_ranges.items():
            v = c.get(name)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                if v < lo or v > hi:
                    result.warnings.append(
                        f"{FEATURE_SPECS[name]['label']} = {v:g} lies outside the range seen "
                        f"during training ({lo:g} to {hi:g}); the prediction may be unreliable.")
    return result


def load_model_bundle(path: str | Path = MODEL_PATH) -> dict:
    """Load the Joblib bundle {model, features, threshold, metadata}."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(MODEL_HELP.format(path=path))
    bundle = joblib.load(path)
    missing = {"model", "features", "threshold"} - set(bundle)
    if missing:
        raise ValueError(f"Model file is incomplete (missing keys: {sorted(missing)}). Retrain it.")
    if list(bundle["features"]) != FEATURES:
        raise ValueError("Model feature order does not match src/config.py. Retrain the model.")
    return bundle


def predict(bundle: dict, cleaned: dict) -> dict:
    """Predict for ONE validated record. Returns class, probabilities and wording."""
    row = pd.DataFrame([{f: cleaned.get(f, np.nan) for f in FEATURES}], columns=FEATURES)
    prob_disease = float(bundle["model"].predict_proba(row)[0, 1])
    threshold = float(bundle.get("threshold", DECISION_THRESHOLD))
    predicted = int(prob_disease >= threshold)
    return {
        "predicted_class": predicted,
        "class_label": ("Pattern similar to the dataset's liver-patient class (higher likelihood)"
                        if predicted == 1 else
                        "Pattern similar to the dataset's non-liver-patient class (lower likelihood)"),
        "probability_liver_class": prob_disease,
        "probability_non_liver_class": 1.0 - prob_disease,
        "threshold": threshold,
    }
