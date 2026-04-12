Project summary: 
This project continues my research on biomarkers for neurodegenerative diseases. The Alzheimer’s Disease Neuroimaging Initiative (ADNI) research project I am conducting is a multimodal, integrative investigation aimed at characterizing the biological and clinical heterogeneity of Alzheimer’s disease across its full continuum. Leveraging the rich, longitudinal ADNI dataset, this study synthesizes neuropathologic indices (including CERAD neurotic plaque scores and NIA-AA ABC staging), molecular imaging biomarkers (amyloid and FDG PET), biofluid measures (CSF Aβ42 and tau), genetic risk factors (APOE ε4), and detailed cognitive assessments (e.g., MMSE and CDR). By aligning in vivo biomarkers with postmortem neuropathology where available, the project seeks to clarify how distinct pathologic processes relate to cognitive decline and diagnostic trajectories, while accounting for demographic and clinical covariates.

Dataset Description:
The dataset is derived from a multimodal ADNI-based table containing clinical, cognitive, and physiological features.

Dataset Characteristics:
4000 observations
~260 features after encoding
Mixed data types:
clinical scores (e.g., MM tasks)
vitals (e.g., blood pressure, pulse)
behavioral indicators (e.g., GDS features)

Target Variable: y_mmse_binary

Preprocessing:
Removal of data leakages, identifiers, and scores.
Placeholder values replaced with NaN
Undersampling was used to balance the dataset to 50-50
One-hot encoding for XGBoost

Methodology included:
Filtering rows with only valid target labels
Removed leakage features
Used one-hot encoding
Undersampling for a balanced dataset

Models Evaluated:
Random Forest
XGBoost Classifier
SVM

Models were evaluated and validated using a gridsearch and 5 fold cross validation, and evaluated based on Accuracy, Precision, Recall, F1-score, and ROC-AUC.

Feature importance was also taken into account, mostly using SHAP. I found that around 99% of the model performance can be retained while reducing the feature count from 262 to 19.



How to Run the Project
1. Clone Repository
git clone https://github.com/KoleGuenther/AlzheimerResearch.git
cd AlzheimerResearch
2. Install Dependencies
pip install -r requirements.txt
3. Run Training and Evaluation
python testHere/test_model.py
