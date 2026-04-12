from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


def run_baseline(data_path: Path | None = None, *, verbose: bool = True) -> pd.DataFrame:
    if data_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        data_path = repo_root / "testHere" / "data" / "alzheimers_balanced_small.csv"

    df = pd.read_csv(data_path)

    if verbose:
        print("Loaded shape:", df.shape)
        print(df["y_mmse_binary"].value_counts())

    X = df.drop(columns=["y_mmse_binary"])
    y = df["y_mmse_binary"].astype(int)

    X_enc = pd.get_dummies(X, drop_first=False, dummy_na=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc,
        y,
        stratify=y,
        random_state=42,
        test_size=0.3,
    )

    def evaluate_model(name: str, model) -> dict:
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
            "PR-AUC": average_precision_score(y_test, y_prob),
        }

        if verbose:
            print(f"\n{name} Results:")
            for k, v in metrics.items():
                if k != "Model":
                    print(f"{k}: {v:.4f}")

        return metrics

    rf = RandomForestClassifier(random_state=42)
    rf_grid = {
        "n_estimators": [100],
        "max_depth": [None, 10],
    }
    rf_search = GridSearchCV(
        rf,
        rf_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_
    rf_metrics = evaluate_model("Random Forest", rf_best)

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    xgb_grid = {
        "n_estimators": [100],
        "max_depth": [3, 5],
        "learning_rate": [0.1],
    }
    xgb_search = GridSearchCV(
        xgb,
        xgb_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
    )
    xgb_search.fit(X_train, y_train)
    xgb_best = xgb_search.best_estimator_
    xgb_metrics = evaluate_model("XGBoost", xgb_best)

    svm_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", SVC(probability=True, random_state=42)),
    ])
    svm_grid = {
        "svm__kernel": ["linear", "rbf"],
        "svm__C": [1],
    }
    svm_search = GridSearchCV(
        svm_pipeline,
        svm_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
    )
    svm_search.fit(X_train, y_train)
    svm_best = svm_search.best_estimator_
    svm_metrics = evaluate_model("SVM", svm_best)

    results_df = pd.DataFrame([rf_metrics, xgb_metrics, svm_metrics])

    if verbose:
        print("\nFinal Results:")
        print(results_df)

    return results_df


def main() -> None:
    results_df = run_baseline(verbose=True)
    assert results_df["ROC-AUC"].max() > 0.7, "Model performance too low"


if __name__ == "__main__":
    main()
