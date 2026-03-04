from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, cast

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

@dataclass
class ModelResult:
    """Container for model evaluation metrics."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    tn: int
    fp: int
    fn: int
    tp: int

class BreastCancerModelComparator:
    """Train and compare multiple classifiers on the breast cancer dataset."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.feature_names: list[str] = []
        self.target_names: list[str] = []
        self.X: pd.DataFrame | None = None
        self.y: pd.Series | None = None
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None
        self.models: Dict[str, Pipeline] = {}
        self.results: list[ModelResult] = []

    def load_data(self) -> None:
        """Load dataset features and target labels."""
        dataset = cast(Any, load_breast_cancer(as_frame=True))
        self.X = dataset.data
        self.y = dataset.target
        self.feature_names = list(dataset.feature_names)
        self.target_names = list(dataset.target_names)

    def split_data(self) -> None:
        """Split dataset into train/test using stratification."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y
        )

    def build_models(self) -> None:
        """Create three classification pipelines."""
        self.models = {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, random_state=self.random_state)),
            ]),
            "Support Vector Machine": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(probability=True, random_state=self.random_state)),
            ]),
            "Random Forest": Pipeline([("clf", RandomForestClassifier(
                n_estimators=300, random_state=self.random_state, class_weight="balanced"
            ))]),
        }

    def evaluate_models(self) -> pd.DataFrame:
        """Train each model and compute evaluation metrics."""
        if not self.models:
            raise ValueError("Models are not built. Call build_models() first.")
        if any(v is None for v in (self.X_train, self.X_test, self.y_train, self.y_test)):
            raise ValueError("Data is not split. Call split_data() first.")

        self.results = []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        X_train, X_test = self.X_train, self.X_test
        y_train, y_test = self.y_train, self.y_test
        assert X_train is not None and X_test is not None and y_train is not None and y_test is not None

        for model_name, pipeline in self.models.items():
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            accuracy, precision, recall, f1, roc_auc = (
                float(accuracy_score(y_test, y_pred)),
                float(precision_score(y_test, y_pred)),
                float(recall_score(y_test, y_pred)),
                float(f1_score(y_test, y_pred)),
                float(roc_auc_score(y_test, y_prob)),
            )
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
            )
            self.results.append(
                ModelResult(
                    model_name=model_name,
                    accuracy=accuracy,
                precision=precision,
                    recall=recall,
                    f1=f1,
                    roc_auc=roc_auc,
                    cv_accuracy_mean=float(cv_scores.mean()),
                    cv_accuracy_std=float(cv_scores.std()),
                    tn=int(tn),
                    fp=int(fp),
                    fn=int(fn),
                    tp=int(tp),
                )
            )
        results_df = pd.DataFrame([result.__dict__ for result in self.results])
        results_df = results_df.sort_values(by=["f1", "roc_auc", "accuracy"], ascending=False)
        return results_df

    def run(self) -> pd.DataFrame:
        """End-to-end execution: load, split, build, evaluate."""
        self.load_data()
        self.split_data()
        self.build_models()
        return self.evaluate_models()

def main() -> None:
    comparator = BreastCancerModelComparator(test_size=0.2, random_state=42)
    result_table = comparator.run()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    print("\nBreast Cancer Classification Model Comparison\n")
    print(result_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if __name__ == "__main__":
    main()
