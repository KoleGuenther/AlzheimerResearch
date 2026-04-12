# AlzheimerResearch

Project summary: 
This project continues my research on biomarkers for neurodegenerative diseases. The Alzheimer’s Disease Neuroimaging Initiative (ADNI) research project I am conducting is a multimodal, integrative investigation aimed at characterizing the biological and clinical heterogeneity of Alzheimer’s disease across its full continuum. Leveraging the rich, longitudinal ADNI dataset, this study synthesizes neuropathologic indices (including CERAD neurotic plaque scores and NIA-AA ABC staging), molecular imaging biomarkers (amyloid and FDG PET), biofluid measures (CSF Aβ42 and tau), genetic risk factors (APOE ε4), and detailed cognitive assessments (e.g., MMSE and CDR). By aligning in vivo biomarkers with postmortem neuropathology where available, the project seeks to clarify how distinct pathologic processes relate to cognitive decline and diagnostic trajectories, while accounting for demographic and clinical covariates.

## Dataset Description
The dataset is derived from a multimodal ADNI-based table containing clinical, cognitive, and physiological features.

### Dataset Characteristics
- ~4,000 observations in the full workflow
- ~260 features after encoding (full workflow)
- Mixed data types:
  - clinical scores (e.g., MM tasks)
  - vitals (e.g., blood pressure, pulse)
  - behavioral indicators (e.g., GDS features)
- Target variable: `y_mmse_binary`

### Preprocessing
- Removal of leakage features, identifiers, and scores
- Placeholder values replaced with `NaN`
- One-hot encoding used for tree/linear models
- Undersampling used to create a balanced dataset variant

## Modeling Approach
Models evaluated across notebooks include:
- Random Forest
- XGBoost Classifier
- SVM

Evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision (PR-AUC)

Feature importance was also explored (including SHAP in the `notebooks/shap` workflow), with strong performance retained under reduced feature sets.

---

## Scalability and Deployment
This baseline workflow can scale from local experimentation to production by packaging the training logic in `src/alzheimer_research/train_baseline.py` and parameterizing data/model settings via config files or CLI arguments. For larger datasets, you can swap local CSV loading for a feature store or cloud object storage, parallelize hyperparameter search, and run scheduled retraining jobs in containers (Docker) on platforms like AWS Batch, Vertex AI, or Azure ML. The same package structure also makes it straightforward to deploy inference as a lightweight API service (e.g., FastAPI) with model artifacts versioned in an experiment tracker or model registry.

## Running the project (updated for `src/`)
The quick-start model run now lives in:
- Script: `src/alzheimer_research/train_baseline.py`
- Compatibility wrapper script (legacy path): `testHere/test_model.py`
- Notebook (legacy, same logic in one cell): `notebooks/modeling/test_model.ipynb`
- Input data: `testHere/data/alzheimers_balanced_small.csv`

The Python script uses a repo-relative default data path, so you can run it directly from the repo root.

### 1) Clone and enter the repo
```bash
git clone https://github.com/KoleGuenther/AlzheimerResearch.git
cd AlzheimerResearch
```

### 2) Create and activate a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows PowerShell
```

### 3) Install dependencies
```bash
pip install -U pip
pip install -r requirements.txt
```

### 4) Run the quick-start model script
```bash
python src/alzheimer_research/train_baseline.py
```
This runs the same model comparison workflow (Random Forest, XGBoost, and SVM) and prints metrics/results in terminal.

### 5) (Optional) Run the notebook version
```bash
jupyter notebook notebooks/modeling/test_model.ipynb
```
Then use **Run All**.
