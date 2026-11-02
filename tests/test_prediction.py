"""Tests for input validation and prediction behaviour.

A tiny model is trained on SYNTHETIC data inside the test (never on real patient data)
so the tests run without the dataset or a trained model file.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from src.config import DECISION_THRESHOLD, FEATURES
from src.prediction import load_model_bundle, predict, validate_inputs
from src.preprocessing import make_pipeline

VALID = {"Age": 45, "Gender": "Male", "Total_Bilirubin": 1.0, "Direct_Bilirubin": 0.3,
         "Alkaline_Phosphotase": 200, "Alamine_Aminotransferase": 30,
         "Aspartate_Aminotransferase": 35, "Total_Proteins": 6.8, "Albumin": 3.5,
         "Albumin_Globulin_Ratio": 1.0}


def tiny_bundle():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n)
    X = pd.DataFrame({
        "Age": rng.integers(20, 80, n), "Gender": rng.choice(["Male", "Female"], n),
        "Total_Bilirubin": np.exp(rng.normal(0.3 * y, 0.5)), "Direct_Bilirubin": np.exp(rng.normal(-1 + 0.3 * y, 0.5)),
        "Alkaline_Phosphotase": np.exp(rng.normal(5 + 0.3 * y, 0.3)),
        "Alamine_Aminotransferase": np.exp(rng.normal(3 + 0.6 * y, 0.5)),
        "Aspartate_Aminotransferase": np.exp(rng.normal(3 + 0.6 * y, 0.5)),
        "Total_Proteins": rng.normal(6.5, 0.7, n), "Albumin": rng.normal(3.3, 0.4, n),
        "Albumin_Globulin_Ratio": rng.normal(1, 0.2, n)})[FEATURES]
    model = CalibratedClassifierCV(make_pipeline(LogisticRegression(max_iter=500)), cv=3).fit(X, y)
    return {"model": model, "features": FEATURES, "threshold": DECISION_THRESHOLD, "metadata": {}}


class TestValidation(unittest.TestCase):
    def test_valid_input_passes(self):
        r = validate_inputs(VALID)
        self.assertTrue(r.is_valid, r.errors)
        self.assertEqual(r.cleaned["Gender"], "Male")

    def test_string_numbers_are_accepted(self):
        r = validate_inputs({**VALID, "Age": "52", "Albumin": "3,4"})
        self.assertTrue(r.is_valid, r.errors)
        self.assertAlmostEqual(r.cleaned["Albumin"], 3.4)

    def test_non_numeric_rejected(self):
        r = validate_inputs({**VALID, "Age": "abc"})
        self.assertFalse(r.is_valid)

    def test_out_of_range_rejected(self):
        self.assertFalse(validate_inputs({**VALID, "Age": 500}).is_valid)
        self.assertFalse(validate_inputs({**VALID, "Total_Bilirubin": -2}).is_valid)

    def test_required_field_missing(self):
        bad = dict(VALID)
        bad["Albumin"] = None
        self.assertFalse(validate_inputs(bad).is_valid)

    def test_optional_ag_ratio_can_be_blank(self):
        r = validate_inputs({**VALID, "Albumin_Globulin_Ratio": ""})
        self.assertTrue(r.is_valid)
        self.assertTrue(np.isnan(r.cleaned["Albumin_Globulin_Ratio"]))
        self.assertTrue(r.warnings)

    def test_invalid_gender_rejected(self):
        self.assertFalse(validate_inputs({**VALID, "Gender": "Robot"}).is_valid)

    def test_cross_field_checks(self):
        self.assertFalse(validate_inputs({**VALID, "Direct_Bilirubin": 5.0}).is_valid)
        self.assertFalse(validate_inputs({**VALID, "Albumin": 7.0}).is_valid)

    def test_nan_and_inf_rejected(self):
        self.assertFalse(validate_inputs({**VALID, "Age": float("inf")}).is_valid)
        self.assertFalse(validate_inputs({**VALID, "Age": float("nan")}).is_valid)

    def test_out_of_training_range_warns_only(self):
        r = validate_inputs(VALID, training_ranges={"Age": (60.0, 90.0)})
        self.assertTrue(r.is_valid)
        self.assertTrue(any("outside the range" in w for w in r.warnings))


class TestPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = tiny_bundle()

    def test_prediction_structure_and_probabilities(self):
        out = predict(self.bundle, validate_inputs(VALID).cleaned)
        self.assertIn(out["predicted_class"], (0, 1))
        self.assertAlmostEqual(out["probability_liver_class"] + out["probability_non_liver_class"], 1.0)
        self.assertTrue(0.0 <= out["probability_liver_class"] <= 1.0)

    def test_missing_optional_value_still_predicts(self):
        cleaned = validate_inputs({**VALID, "Albumin_Globulin_Ratio": None}).cleaned
        self.assertIn("predicted_class", predict(self.bundle, cleaned))

    def test_extreme_values_raise_probability(self):
        high = {**VALID, "Total_Bilirubin": 20, "Direct_Bilirubin": 10,
                "Alamine_Aminotransferase": 400, "Aspartate_Aminotransferase": 400}
        p_norm = predict(self.bundle, validate_inputs(VALID).cleaned)["probability_liver_class"]
        p_high = predict(self.bundle, validate_inputs(high).cleaned)["probability_liver_class"]
        self.assertGreater(p_high, p_norm)

    def test_missing_model_file_message(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_model_bundle("no/such/model.joblib")
        self.assertIn("python train_model.py", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
