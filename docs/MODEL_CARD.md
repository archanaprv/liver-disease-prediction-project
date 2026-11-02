# Model Card: ILPD Educational Classifier

## Summary

This project trains a binary classifier on the Indian Liver Patient Dataset (ILPD) and exposes its dataset-class probabilities through a Streamlit interface. It is intended to demonstrate a machine-learning workflow, not to provide a health assessment.

## Intended use

**Appropriate use:** classroom assignments, software demonstrations, and learning about tabular-data preprocessing and model evaluation.

**Not appropriate:** medical diagnosis, screening, triage, treatment decisions, or communicating that a person does or does not have liver disease.

The class names refer only to labels in the source dataset:

- `1`: source dataset liver-patient class (original ILPD label `1`)
- `0`: source dataset non-liver-patient class (original ILPD label `2`)

They are not verified clinical diagnoses.

## Data

- **Dataset:** Indian Liver Patient Dataset (ILPD), included as `data/indian_liver_patient.csv`.
- **Source information:** [UCI Machine Learning Repository: ILPD](https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset).
- **Dataset license listed by UCI:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Citation:** Ramana, B. & Venkateswarlu, N. (2022). *ILPD (Indian Liver Patient Dataset) [Dataset].* UCI Machine Learning Repository. https://doi.org/10.24432/C5D02C.
- **Included dataset size:** 583 records before duplicate removal, with 10 input features and one target label.
- **Known limitations:** small sample, class imbalance, one source population, no independent external validation, and age/gender fields that UCI identifies as potentially sensitive.

The included CSV's structure matches the headerless UCI format; its file contents have not been independently verified byte-for-byte against the repository download. Dataset terms are separate from the license for this project's code.

## Method

1. Load and normalize supported headered or headerless ILPD CSV formats; map source labels to `1` and `0`.
2. Remove exact duplicate rows by default.
3. Use a stratified 80/20 train-test split with random seed `42`.
4. On the training split only, use stratified five-fold grid search to compare Logistic Regression, Decision Tree, Random Forest, and RBF-kernel SVM pipelines. Select by mean cross-validated ROC-AUC; use cross-validated F1 to break ties.
5. Apply sigmoid probability calibration to the selected estimator using the training split.
6. Evaluate that calibrated estimator once on the held-out test split. Calculate bootstrap 95% intervals for ROC-AUC, recall, precision, and F1.

Preprocessing is included in each scikit-learn pipeline: median imputation and scaling for numeric features, log transformation for selected skewed lab values, and most-frequent imputation plus one-hot encoding for gender.

## Evaluation snapshot

The current seeded run selected the RBF SVM. The test split contained 114 records: 81 from the positive (liver-patient) dataset class and 33 from the negative class.

| Metric | Held-out test result | Bootstrap 95% interval |
|---|---:|---:|
| ROC-AUC | 0.815 | 0.732–0.888 |
| Recall / sensitivity (class `1`) | 0.926 | 0.864–0.978 |
| Precision (class `1`) | 0.750 | 0.660–0.833 |
| F1 (class `1`) | 0.829 | 0.766–0.881 |
| Specificity (class `0`) | 0.242 | Not calculated |
| Accuracy | 0.728 | Not calculated |

At the default decision threshold of `0.5`, the confusion counts were 8 true negatives, 25 false positives, 6 false negatives, and 75 true positives. The low specificity is an important limitation: most negative-class records in this small test split were incorrectly labeled positive. These figures are a single split's results and should not be generalized to other populations or interpreted as clinical performance.

## Limitations and risks

- There is no independent or external validation.
- The data is small, class-imbalanced, and from one source population; distribution shift is likely across settings and populations.
- Source labels and measurements may have limitations or errors.
- Ten routine variables cannot represent the full clinical context needed to assess liver disease.
- Predicted probabilities are model scores learned from this dataset; they are not an individual's clinical risk.
- A high recall at the selected threshold comes with poor specificity in this test split.

## Maintenance and reproducibility

Training configuration, generated metrics, plots, and the saved model are produced by `python train_model.py`. The generated model and report files are intentionally not committed by default. Dependency requirements currently specify minimum versions rather than a fully locked environment, so exact reproducibility across environments is not guaranteed.

## Further evaluation needed

Before any claims beyond educational demonstration, use appropriately licensed independent data, define a representative validation plan, assess calibration and errors across relevant subgroups, and have qualified domain experts review the intended use and results.
