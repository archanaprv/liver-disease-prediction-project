# Liver Disease Prediction Using Machine Learning

B.Tech Artificial Intelligence and Data Science minor project (KMCT Institute of Technology and Management).

> **Educational prototype only.** This application is not a medical device, does not diagnose liver
> disease and must not replace a qualified healthcare professional.

It trains and compares four classifiers (Logistic Regression, Decision Tree, Random Forest, SVM) on the
Indian Liver Patient Dataset (ILPD), saves the best calibrated pipeline with Joblib and serves it
through a Streamlit web app with input validation.

## Project structure
```
liver-disease-prediction/
  app.py               Streamlit interface
  train_model.py       Training, comparison, evaluation, saving
  requirements.txt
  data/                Put indian_liver_patient.csv here
  models/              liver_model.joblib is created here
  outputs/             Metrics, tables and plots are created here
  src/config.py        Paths, feature order, target, input ranges
  src/preprocessing.py Dataset loader, data-quality report, preprocessing pipeline
  src/evaluation.py    Metrics, bootstrap CI, plots
  src/prediction.py    Input validation, model loading, prediction
  tests/               Unit tests
```

## 1. Setup on Windows (VS Code)
Install Python 3.10+ from python.org (tick "Add Python to PATH"). Open the project folder in VS Code
(File > Open Folder), then open a terminal (Terminal > New Terminal, PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
If PowerShell blocks activation, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use Command Prompt: `venv\Scripts\activate.bat`.
In VS Code press Ctrl+Shift+P > "Python: Select Interpreter" > choose the `venv` interpreter.

## 2. Dataset
Download ILPD from one of the sources below, check its licence/citation terms on the page, and save it as
`data/indian_liver_patient.csv`:
- UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset
- Kaggle copy named `indian_liver_patient.csv`

Both the headerless UCI file and the Kaggle CSV (with header) are supported. Original coding:
label 1 = liver patient, label 2 = non-liver patient. The code maps this to
`Liver_Disease` = 1 / 0. Use `--data <path>` if the file is stored elsewhere.

## 3. Train
```powershell
python train_model.py
```
This prints the dataset summary (records, missing values, duplicates, class counts) and writes to `outputs/`:
`data_summary.json`, `metrics.json`, `model_comparison.csv`, `classification_report.txt`,
`test_predictions.csv`, `feature_importance.csv` and PNG plots (class distribution, feature distributions,
correlation heatmap, confusion matrices, ROC curves, model comparison, calibration, feature importance).
It also writes `models/liver_model.joblib`. Use **these generated numbers and plots** in the report and slides.

## 4. Run the app
```powershell
streamlit run app.py
```
Open http://localhost:8501 if the browser does not open automatically.

## 5. Test
```powershell
python -m unittest discover -s tests -v
```
(`pytest` also works if installed.) Tests need neither the dataset nor a trained model.

Try in the app: valid values (defaults); Age = 500 (rejected); Direct bilirubin greater than total (rejected);
untick "A/G ratio is known" (median imputed with a warning).

## Evaluation protocol
Exact duplicates removed; stratified 80/20 split (seed 42); preprocessing inside a scikit-learn Pipeline so
statistics are learned from training data only; 5-fold stratified grid search on the training set; model
selected by cross-validated ROC-AUC; sigmoid calibration on training data; one final evaluation on the
held-out test set, with bootstrap 95% confidence intervals in `metrics.json`.

## Troubleshooting
| Problem | Fix |
|---|---|
| `Dataset file not found` | Place the CSV at `data/indian_liver_patient.csv` or pass `--data`. |
| `Trained model not found` in the app | Run `python train_model.py` first. |
| `streamlit is not recognized` | Activate the venv, then `pip install -r requirements.txt`. |
| Model fails to load after upgrading libraries | Re-run `python train_model.py` (the saved file is version-specific). |
| Training takes a few minutes | Normal: the Random Forest grid search is the slowest step. |
