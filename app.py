
"""Streamlit interface for the Liver Disease Prediction educational prototype.

Run: streamlit run app.py
All validation and prediction logic lives in src/prediction.py (unit-tested).
"""

import json

import streamlit as st

from src.config import FEATURE_SPECS, GENDER_VALUES, MODEL_PATH, OUTPUTS_DIR
from src.prediction import (
    DISCLAIMER,
    load_model_bundle,
    predict,
    validate_inputs,
)

st.set_page_config(
    page_title="Liver Disease Prediction (Educational)",
    page_icon="🧪",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_bundle():
    return load_model_bundle(MODEL_PATH)


def show_images(names):
    """Display available model evaluation images."""
    cols = st.columns(2)
    shown = False

    for i, (title, fname) in enumerate(names):
        path = OUTPUTS_DIR / fname

        if path.exists():
            cols[i % 2].image(
                str(path),
                caption=title,
                use_container_width=True,
            )
            shown = True

    return shown


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Liver Disease Prediction Using Machine Learning")
st.caption(
    "B.Tech AI & Data Science minor project - educational prototype"
)
st.warning(DISCLAIMER, icon="⚠️")


# ---------------------------------------------------------------------------
# Load trained model
# ---------------------------------------------------------------------------

try:
    bundle = get_bundle()
except (FileNotFoundError, ValueError) as exc:
    st.error("The trained model could not be loaded.")
    st.code(str(exc))
    st.stop()

meta = bundle.get("metadata", {})
training_ranges = meta.get("training_ranges")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_predict, tab_perf, tab_about = st.tabs(
    ["Prediction", "Model performance", "About & limitations"]
)


# ---------------------------------------------------------------------------
# Prediction tab
# ---------------------------------------------------------------------------

with tab_predict:
    st.subheader("Enter clinical parameters")

    st.write(
        "Values are checked before prediction. Ranges are plausibility "
        "limits for catching typing errors, not clinical reference ranges."
    )

    with st.form("input_form"):
        c1, c2, c3 = st.columns(3)
        columns = [c1, c2, c3]
        raw = {}

        for i, (name, spec) in enumerate(FEATURE_SPECS.items()):
            col = columns[i % 3]

            # Gender selection
            if name == "Gender":
                raw[name] = col.selectbox(
                    "Gender",
                    GENDER_VALUES,
                    help=spec["desc"],
                )

            # Age input as an integer
            elif name == "Age":
                raw[name] = col.number_input(
                    f"{spec['label']}"
                    + (f" ({spec['unit']})" if spec["unit"] else ""),
                    min_value=int(spec["min"]),
                    max_value=int(spec["max"]),
                    value=int(spec["default"]),
                    step=1,
                    help=(
                        f"{spec['desc']} Accepted: "
                        f"{int(spec['min'])} to "
                        f"{int(spec['max'])} years."
                    ),
                )

            # Albumin/Globulin ratio with optional missing value
            elif name == "Albumin_Globulin_Ratio":
                known = col.checkbox(
                    "A/G ratio is known",
                    value=True,
                    help=(
                        "Untick if unavailable; the median is imputed."
                    ),
                )

                value = col.number_input(
                    f"{spec['label']}",
                    min_value=0.0,
                    max_value=float(spec["max"]) * 2,
                    value=float(spec["default"]),
                    step=float(spec["step"]),
                    help=spec["desc"],
                )

                raw[name] = value if known else None

            # Other numerical inputs
            else:
                unit = (
                    f" ({spec['unit']})"
                    if spec["unit"]
                    else ""
                )

                raw[name] = col.number_input(
                    f"{spec['label']}{unit}",
                    value=float(spec["default"]),
                    step=float(spec["step"]),
                    help=(
                        f"{spec['desc']} Accepted: "
                        f"{spec['min']:g} to {spec['max']:g}."
                    ),
                )

        submitted = st.form_submit_button(
            "Estimate",
            type="primary",
        )

    # Process prediction
    if submitted:
        check = validate_inputs(raw, training_ranges)

        if not check.is_valid:
            st.error(
                "Please correct the following before a prediction can be made:"
            )

            for msg in check.errors:
                st.markdown(f"- {msg}")

        else:
            result = predict(bundle, check.cleaned)

            for msg in check.warnings:
                st.warning(msg)

            st.subheader("Result")

            if result["predicted_class"] == 1:
                st.error(result["class_label"])
            else:
                st.success(result["class_label"])

            m1, m2 = st.columns(2)

            m1.metric(
                "Estimated probability of liver-patient class",
                f"{100 * result['probability_liver_class']:.1f}%",
            )

            m2.metric(
                "Estimated probability of non-liver-patient class",
                f"{100 * result['probability_non_liver_class']:.1f}%",
            )

            st.progress(
                min(
                    max(result["probability_liver_class"], 0.0),
                    1.0,
                )
            )

            st.caption(
                f"Decision threshold: {result['threshold']:.2f}. "
                "Probabilities are calibrated on the training data, "
                "but the dataset is small and comes from a single region, "
                "so treat them as rough scores, not clinical risks."
            )

            st.info(DISCLAIMER)


# ---------------------------------------------------------------------------
# Model performance tab
# ---------------------------------------------------------------------------

with tab_perf:
    st.subheader("Evaluation on the held-out test set")

    metrics_file = OUTPUTS_DIR / "metrics.json"

    if not metrics_file.exists():
        st.info(
            "No evaluation results found. Run `python train_model.py` "
            "to generate them."
        )

    else:
        metrics = json.loads(metrics_file.read_text())
        final = metrics["final_test_metrics_calibrated"]

        st.write(
            f"Selected model: **{metrics['selected_model']}** "
            f"(calibrated). "
            f"Test set size: {metrics['protocol']['n_test']} records."
        )

        cols = st.columns(5)

        metric_items = [
            ("Accuracy", "accuracy"),
            ("Precision", "precision"),
            ("Recall", "recall_sensitivity"),
            ("F1", "f1"),
            ("ROC-AUC", "roc_auc"),
        ]

        for c, (label, key) in zip(cols, metric_items):
            c.metric(label, f"{final[key]:.3f}")

        st.caption(
            "Small test set: see the bootstrap confidence intervals in "
            "outputs/metrics.json before drawing conclusions."
        )

        show_images(
            [
                ("Confusion matrix", "confusion_matrix.png"),
                ("ROC curves", "roc_curves.png"),
                ("Model comparison", "model_comparison.png"),
                ("Feature importance", "feature_importance.png"),
            ]
        )

        csv = OUTPUTS_DIR / "model_comparison.csv"

        if csv.exists():
            import pandas as pd

            st.dataframe(
                pd.read_csv(csv),
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# About and limitations tab
# ---------------------------------------------------------------------------

with tab_about:
    st.subheader("Purpose")

    st.write(
        "This project demonstrates how a supervised machine-learning "
        "pipeline can learn patterns from a historical labelled dataset "
        "(the Indian Liver Patient Dataset, ILPD) and present a "
        "probability-style score through a web interface."
    )

    st.subheader("Method")

    st.markdown(
        "- Median / most-frequent imputation, log transform of skewed "
        "lab values, scaling, one-hot encoding\n"
        "- Four classifiers compared with 5-fold stratified "
        "cross-validation on the training split\n"
        "- Final model chosen by cross-validated ROC-AUC, then "
        "probability-calibrated\n"
        "- One pipeline object is saved with Joblib and reused "
        "unchanged by this app"
    )

    st.subheader("Limitations")

    st.markdown(
        "- Small dataset (a few hundred records) from one region; "
        "no external validation\n"
        "- Class labels are the dataset's own and may contain errors; "
        "the classes are imbalanced\n"
        "- Only ten routine parameters are used; many liver conditions "
        "need imaging and clinical review\n"
        "- Performance on other populations, labs or measurement units "
        "is unknown\n"
        "- **Not a diagnostic tool and not a substitute for a qualified "
        "healthcare professional**"
    )

    if meta:
        st.caption(
            f"Model: {meta.get('model_name')} | "
            f"trained {meta.get('trained_at')} | "
            f"scikit-learn {meta.get('sklearn_version')}"
        )
