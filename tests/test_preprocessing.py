"""Tests for dataset loading and the preprocessing pipeline (uses tiny in-memory data)."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import FEATURES, TARGET
from src.preprocessing import build_preprocessor, data_quality_report, load_dataset

UCI_ROWS = [
    "65,Female,0.7,0.1,187,16,18,6.8,3.3,0.9,1",
    "62,Male,10.9,5.5,699,64,100,7.5,3.2,0.74,1",
    "20,Male,1.1,0.5,128,20,30,3.9,1.9,,2",
    "45,Female,0.8,0.2,210,25,28,6.5,3.4,1.1,2",
]


class TestLoading(unittest.TestCase):
    def _write(self, text):
        d = tempfile.mkdtemp()
        p = Path(d) / "ilpd.csv"
        p.write_text(text)
        return p

    def test_headerless_uci_file(self):
        df = load_dataset(self._write("\n".join(UCI_ROWS)))
        self.assertEqual(list(df.columns), FEATURES + [TARGET])
        self.assertEqual(df[TARGET].tolist(), [1, 1, 0, 0])  # label 2 -> 0
        self.assertEqual(int(df["Albumin_Globulin_Ratio"].isna().sum()), 1)

    def test_kaggle_header_names(self):
        header = ("Age,Gender,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,"
                  "Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,"
                  "Albumin,Albumin_and_Globulin_Ratio,Dataset")
        df = load_dataset(self._write(header + "\n" + "\n".join(UCI_ROWS)))
        self.assertEqual(list(df.columns), FEATURES + [TARGET])

    def test_missing_file_gives_helpful_message(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_dataset("does/not/exist.csv")
        self.assertIn("Dataset file not found", str(ctx.exception))

    def test_invalid_values_become_nan(self):
        rows = UCI_ROWS + ["30,Male,-1,0.2,100,20,20,6,3,1,1"]
        df = load_dataset(self._write("\n".join(rows)))
        self.assertTrue(np.isnan(df.loc[4, "Total_Bilirubin"]))
        self.assertGreaterEqual(df.attrs["invalid_values_set_to_nan"], 1)

    def test_zero_direct_bilirubin_is_valid(self):
        rows = UCI_ROWS + ["30,Male,1,0,100,20,20,6,3,1,1"]
        df = load_dataset(self._write("\n".join(rows)))
        self.assertEqual(df.loc[4, "Direct_Bilirubin"], 0)
        self.assertEqual(df.attrs["invalid_values_set_to_nan"], 0)

    def test_quality_report_counts(self):
        rep = data_quality_report(load_dataset(self._write("\n".join(UCI_ROWS))))
        self.assertEqual(rep["n_records"], 4)
        self.assertEqual(rep["class_counts"]["liver_patient_1"], 2)
        self.assertEqual(rep["missing_values_per_column"]["Albumin_Globulin_Ratio"], 1)


class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        n = 60
        self.df = pd.DataFrame({
            "Age": rng.integers(20, 80, n), "Gender": rng.choice(["Male", "Female"], n),
            "Total_Bilirubin": rng.uniform(0.5, 10, n), "Direct_Bilirubin": rng.uniform(0.1, 4, n),
            "Alkaline_Phosphotase": rng.uniform(100, 800, n),
            "Alamine_Aminotransferase": rng.uniform(10, 300, n),
            "Aspartate_Aminotransferase": rng.uniform(10, 300, n),
            "Total_Proteins": rng.uniform(5, 8, n), "Albumin": rng.uniform(2, 4.5, n),
            "Albumin_Globulin_Ratio": rng.uniform(0.5, 1.8, n)})
        self.df.loc[3, "Albumin_Globulin_Ratio"] = np.nan

    def test_output_has_no_nan_and_expected_width(self):
        out = build_preprocessor().fit_transform(self.df[FEATURES])
        self.assertFalse(np.isnan(out).any())
        self.assertEqual(out.shape, (60, 10))  # 9 numeric + 1 binary gender column

    def test_statistics_learned_from_train_only(self):
        train, test = self.df.iloc[:40], self.df.iloc[40:]
        prep = build_preprocessor().fit(train[FEATURES])
        scaler = prep.named_transformers_["numeric"].named_steps["scale"]
        self.assertAlmostEqual(float(scaler.mean_[0]), float(train["Age"].mean()), places=6)
        self.assertNotAlmostEqual(float(scaler.mean_[0]), float(self.df["Age"].mean()), places=3)
        self.assertEqual(prep.transform(test[FEATURES]).shape[0], 20)

    def test_unseen_gender_does_not_crash(self):
        prep = build_preprocessor().fit(self.df[FEATURES])
        odd = self.df[FEATURES].iloc[:2].copy()
        odd["Gender"] = ["Other", np.nan]
        self.assertEqual(prep.transform(odd).shape[0], 2)


if __name__ == "__main__":
    unittest.main()
