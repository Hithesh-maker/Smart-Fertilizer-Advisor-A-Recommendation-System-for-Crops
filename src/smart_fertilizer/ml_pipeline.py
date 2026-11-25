"""Leakage-free scikit-learn ML training, OOD distance calibration, and inference pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from smart_fertilizer.domain import SoilReading, normalize_crop, normalize_fertilizer, normalize_soil

NUMERIC_FEATURES = ["Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
CATEGORICAL_FEATURES = ["Soil_Type", "Crop_Type"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "fertilizer_dataset.csv"
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "model_bundle.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics.json"


def load_and_clean_dataset(csv_path: Path | str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load dataset, fix typoed column names, and normalize categorical strings."""
    df = pd.read_csv(csv_path)
    df = df.rename(
        columns={
            "Temparature": "Temperature",
            "Humidity ": "Humidity",
            "Soil Type": "Soil_Type",
            "Crop Type": "Crop_Type",
            "Fertilizer Name": "Fertilizer",
        }
    )
    df = df.dropna(subset=FEATURE_COLUMNS + ["Fertilizer"]).copy()
    df["Soil_Type"] = df["Soil_Type"].map(normalize_soil)
    df["Crop_Type"] = df["Crop_Type"].map(normalize_crop)
    df["Fertilizer"] = df["Fertilizer"].map(normalize_fertilizer)
    return df


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def _aggregate_feature_importance(pipeline: Pipeline) -> list[dict[str, float | str]]:
    estimator = pipeline.named_steps["model"]
    preprocessor: ColumnTransformer = pipeline.named_steps["prep"]
    if not hasattr(estimator, "feature_importances_"):
        return []

    importances = estimator.feature_importances_
    names = preprocessor.get_feature_names_out()
    grouped: dict[str, float] = {col: 0.0 for col in FEATURE_COLUMNS}
    for feat_name, score in zip(names, importances):
        if feat_name.startswith("num__"):
            raw = feat_name[len("num__") :]
            grouped[raw] = grouped.get(raw, 0.0) + float(score)
        elif feat_name.startswith("cat__Soil_Type_"):
            grouped["Soil_Type"] += float(score)
        elif feat_name.startswith("cat__Crop_Type_"):
            grouped["Crop_Type"] += float(score)

    total = sum(grouped.values()) or 1.0
    return [
        {"feature": k, "importance": round((v / total) * 100.0, 2)}
        for k, v in sorted(grouped.items(), key=lambda item: item[1], reverse=True)
    ]


def _build_ood_reference(df: pd.DataFrame) -> dict[str, Any]:
    """Compute standardized numeric reference points & 1-NN leave-one-out distance threshold."""
    unique_df = df.drop_duplicates(subset=NUMERIC_FEATURES).reset_index(drop=True)
    scaler = StandardScaler()
    scaled_unique = scaler.fit_transform(unique_df[NUMERIC_FEATURES])

    # Pairwise Euclidean distance on deduplicated observations (excluding self=0)
    diffs = scaled_unique[:, None, :] - scaled_unique[None, :, :]
    dist_matrix = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    nn_distances = np.min(dist_matrix, axis=1)

    feature_bounds = {
        col: {"min": float(df[col].min()), "max": float(df[col].max())}
        for col in NUMERIC_FEATURES
    }

    return {
        "scaler": scaler,
        "reference_points": scaled_unique,
        "p95_distance": float(np.percentile(nn_distances, 95)),
        "p99_distance": float(np.percentile(nn_distances, 99)),
        "feature_bounds": feature_bounds,
    }


def train_and_evaluate(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    bundle_path: Path | str = DEFAULT_BUNDLE_PATH,
    metrics_path: Path | str = DEFAULT_METRICS_PATH,
) -> dict[str, Any]:
    """Train candidate models on stratified holdout + CV without target leakage."""
    df = load_and_clean_dataset(dataset_path)
    X = df[FEATURE_COLUMNS]
    y = df["Fertilizer"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    candidates = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1500, random_state=42),
    }

    min_class_count = int(y_train.value_counts().min())
    cv_splits = max(2, min(4, min_class_count))
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    df_unique = df.drop_duplicates().reset_index(drop=True)
    X_u = df_unique[FEATURE_COLUMNS]
    y_u = df_unique["Fertilizer"]
    X_u_tr, X_u_te, y_u_tr, y_u_te = train_test_split(X_u, y_u, test_size=0.2, random_state=42)

    benchmarks: list[dict[str, Any]] = []
    trained_pipelines: dict[str, Pipeline] = {}

    for name, estimator in candidates.items():
        pipe = Pipeline([("prep", _build_preprocessor()), ("model", clone(estimator))])
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        test_acc = float(accuracy_score(y_test, preds))
        macro_f1 = float(f1_score(y_test, preds, average="macro"))

        unique_pipe = Pipeline([("prep", _build_preprocessor()), ("model", clone(estimator))])
        unique_pipe.fit(X_u_tr, y_u_tr)
        unique_acc = float(accuracy_score(y_u_te, unique_pipe.predict(X_u_te)))

        trained_pipelines[name] = pipe
        benchmarks.append(
            {
                "model": name,
                "cv_accuracy_mean": round(float(np.mean(cv_scores)) * 100.0, 2),
                "cv_accuracy_std": round(float(np.std(cv_scores)) * 100.0, 2),
                "holdout_accuracy": round(test_acc * 100.0, 2),
                "holdout_macro_f1": round(macro_f1 * 100.0, 2),
                "unique_row_holdout_accuracy": round(unique_acc * 100.0, 2),
            }
        )

    benchmarks.sort(
        key=lambda b: (b["unique_row_holdout_accuracy"], b["holdout_macro_f1"], b["cv_accuracy_mean"]),
        reverse=True,
    )
    best_name = benchmarks[0]["model"]
    best_pipe = trained_pipelines[best_name]
    best_preds = best_pipe.predict(X_test)
    report = classification_report(y_test, best_preds, output_dict=True, zero_division=0)
    feature_importance = _aggregate_feature_importance(best_pipe)

    production_pipe = Pipeline([("prep", _build_preprocessor()), ("model", clone(candidates[best_name]))])
    production_pipe.fit(X, y)
    ood_ref = _build_ood_reference(df)

    metrics_payload = {
        "dataset_rows": int(len(df)),
        "unique_rows": int(len(df_unique)),
        "duplicate_rows_detected": int(df.duplicated().sum()),
        "classes": sorted(y.unique().tolist()),
        "cv_folds": cv_splits,
        "selected_model": best_name,
        "benchmarks": benchmarks,
        "feature_importance": feature_importance,
        "ood_p99_distance": round(ood_ref["p99_distance"], 3),
        "holdout_macro_avg": {
            "precision": round(report["macro avg"]["precision"] * 100.0, 2),
            "recall": round(report["macro avg"]["recall"] * 100.0, 2),
            "f1": round(report["macro avg"]["f1-score"] * 100.0, 2),
        },
    }

    bundle_out = Path(bundle_path)
    bundle_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": production_pipe, "metrics": metrics_payload, "ood": ood_ref}, bundle_out)

    metrics_out = Path(metrics_path)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return metrics_payload


class FertilizerPredictor:
    """Loaded production ML inference wrapper with OOD & low-confidence guardrails."""

    def __init__(self, bundle_path: Path | str = DEFAULT_BUNDLE_PATH) -> None:
        self.bundle_path = Path(bundle_path)
        if not self.bundle_path.exists():
            train_and_evaluate(bundle_path=self.bundle_path)
        loaded = joblib.load(self.bundle_path)
        if "ood" not in loaded:
            train_and_evaluate(bundle_path=self.bundle_path)
            loaded = joblib.load(self.bundle_path)
        self.pipeline: Pipeline = loaded["pipeline"]
        self.metrics: dict[str, Any] = loaded["metrics"]
        self.ood: dict[str, Any] = loaded["ood"]

    def _assess_safety(self, reading: SoilReading, frame: pd.DataFrame, confidence_pct: float) -> dict[str, Any]:
        scaled_query = self.ood["scaler"].transform(frame[NUMERIC_FEATURES])[0]
        dists = np.sqrt(np.sum((self.ood["reference_points"] - scaled_query) ** 2, axis=1))
        min_dist = float(np.min(dists))
        p99 = float(self.ood["p99_distance"])
        ood_threshold = max(2.0, p99 * 1.25)

        out_of_range_fields: list[str] = []
        feat_dict = reading.to_feature_dict()
        for col, bounds in self.ood["feature_bounds"].items():
            val = float(feat_dict[col])
            if val < bounds["min"] or val > bounds["max"]:
                out_of_range_fields.append(f"{col}={val} (trained {bounds['min']}..{bounds['max']})")

        reasons: list[str] = []
        if confidence_pct < 55.0:
            reasons.append(f"Low model confidence ({confidence_pct}% < 55% threshold)")
        if min_dist > ood_threshold or out_of_range_fields:
            detail = f"scaled 1-NN distance {min_dist:.2f} > {ood_threshold:.2f}"
            if out_of_range_fields:
                detail += f"; outside dataset bounds: {', '.join(out_of_range_fields[:3])}"
            reasons.append(f"Extreme/unusual soil profile ({detail})")

        return {
            "low_confidence": confidence_pct < 55.0,
            "ood_flagged": bool(min_dist > ood_threshold or out_of_range_fields),
            "safety_warning": bool(reasons),
            "ood_distance": round(min_dist, 3),
            "ood_threshold": round(ood_threshold, 3),
            "warning_reasons": reasons,
            "safety_message": (
                "Consult local soil testing lab before high-dose application."
                if reasons
                else "Within calibrated dataset operating envelope."
            ),
        }

    def predict(self, reading: SoilReading, top_k: int = 3) -> dict[str, Any]:
        frame = pd.DataFrame([reading.to_feature_dict()])
        probs = self.pipeline.predict_proba(frame)[0]
        classes = self.pipeline.classes_
        ranked_indices = np.argsort(probs)[::-1][:top_k]
        top_predictions = [
            {"fertilizer": str(classes[idx]), "confidence_pct": round(float(probs[idx]) * 100.0, 2)}
            for idx in ranked_indices
        ]
        confidence = top_predictions[0]["confidence_pct"]
        safety = self._assess_safety(reading, frame, confidence)

        return {
            "recommended": top_predictions[0]["fertilizer"],
            "confidence_pct": confidence,
            "top_alternatives": top_predictions,
            "model_used": self.metrics["selected_model"],
            **safety,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and benchmark Smart Fertilizer Advisor models")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    args = parser.parse_args(argv)

    summary = train_and_evaluate(args.dataset, args.bundle, args.metrics)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
