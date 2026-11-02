"""Central configuration shared by training, evaluation, tests and the web app.

Keeping feature names, ranges and paths in ONE place guarantees that training
and deployment use exactly the same feature order and target definition.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "indian_liver_patient.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODEL_PATH = MODELS_DIR / "liver_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
DECISION_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Canonical column names (order matters: this is the model's feature order)
# ---------------------------------------------------------------------------
FEATURES = [
    "Age",
    "Gender",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Proteins",
    "Albumin",
    "Albumin_Globulin_Ratio",
]
CATEGORICAL_FEATURES = ["Gender"]
NUMERIC_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]
# Strongly right-skewed lab values: log1p-transformed before scaling.
LOG_FEATURES = [
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
]
OTHER_NUMERIC_FEATURES = [f for f in NUMERIC_FEATURES if f not in LOG_FEATURES]

TARGET = "Liver_Disease"  # 1 = liver patient class, 0 = non-liver-patient class
GENDER_VALUES = ["Male", "Female"]

# Column order of the headerless UCI file (ILPD)
UCI_COLUMN_ORDER = FEATURES + ["Selector"]

# Normalised (lower-case, alphanumeric only) header -> canonical name.
COLUMN_ALIASES = {
    "age": "Age",
    "gender": "Gender",
    "totalbilirubin": "Total_Bilirubin", "tb": "Total_Bilirubin",
    "directbilirubin": "Direct_Bilirubin", "db": "Direct_Bilirubin",
    "alkalinephosphotase": "Alkaline_Phosphotase",
    "alkalinephosphatase": "Alkaline_Phosphotase",
    "alkphos": "Alkaline_Phosphotase",
    "alamineaminotransferase": "Alamine_Aminotransferase",
    "sgpt": "Alamine_Aminotransferase",
    "aspartateaminotransferase": "Aspartate_Aminotransferase",
    "sgot": "Aspartate_Aminotransferase",
    "totalprotiens": "Total_Proteins", "totalproteins": "Total_Proteins",
    "tp": "Total_Proteins",
    "albumin": "Albumin", "alb": "Albumin",
    "albuminandglobulinratio": "Albumin_Globulin_Ratio",
    "albuminglobulinratio": "Albumin_Globulin_Ratio",
    "agratio": "Albumin_Globulin_Ratio", "ag": "Albumin_Globulin_Ratio",
}
TARGET_ALIASES = {"dataset", "selector", "label", "class", "target", "liverdisease"}

# ---------------------------------------------------------------------------
# Input specification used by validation and by the Streamlit form.
# min/max are broad PLAUSIBILITY limits for catching typing mistakes.
# They are NOT clinical reference ranges.
# ---------------------------------------------------------------------------
FEATURE_SPECS = {
    "Age": dict(label="Age", unit="years", min=1, max=120, default=45, step=1,
                required=True, desc="Age of the person."),
    "Gender": dict(label="Gender", unit="", required=True,
                   desc="Gender as recorded in the dataset (Male / Female)."),
    "Total_Bilirubin": dict(label="Total Bilirubin", unit="mg/dL", min=0.1, max=100.0,
                            default=1.0, step=0.1, required=True,
                            desc="Total bilirubin in blood."),
    "Direct_Bilirubin": dict(label="Direct Bilirubin", unit="mg/dL", min=0.0, max=80.0,
                             default=0.3, step=0.1, required=True,
                             desc="Conjugated (direct) bilirubin."),
    "Alkaline_Phosphotase": dict(label="Alkaline Phosphatase (ALP)", unit="IU/L",
                                 min=10, max=3000, default=200, step=1, required=True,
                                 desc="Enzyme found in liver and bile ducts."),
    "Alamine_Aminotransferase": dict(label="Alanine Aminotransferase (SGPT/ALT)",
                                     unit="IU/L", min=1, max=5000, default=30, step=1,
                                     required=True, desc="Liver enzyme (ALT / SGPT)."),
    "Aspartate_Aminotransferase": dict(label="Aspartate Aminotransferase (SGOT/AST)",
                                       unit="IU/L", min=1, max=5000, default=35, step=1,
                                       required=True, desc="Enzyme (AST / SGOT)."),
    "Total_Proteins": dict(label="Total Proteins", unit="g/dL", min=1.0, max=15.0,
                           default=6.8, step=0.1, required=True,
                           desc="Total protein in blood."),
    "Albumin": dict(label="Albumin", unit="g/dL", min=0.5, max=8.0, default=3.5,
                    step=0.1, required=True, desc="Main protein made by the liver."),
    "Albumin_Globulin_Ratio": dict(label="Albumin/Globulin Ratio", unit="", min=0.1,
                                   max=6.0, default=1.0, step=0.05, required=False,
                                   desc="Optional. If unknown, the pipeline imputes "
                                        "the training-set median."),
}
