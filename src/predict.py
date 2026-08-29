"""Load trained artifacts and predict PD vs HC with explanations."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR
from .explain import explain_prediction


def artifact_path(dataset_key: str, model_name: str = "random_forest") -> Path:
    return MODELS_DIR / f"{dataset_key}__{model_name}.joblib"


def load_model(dataset_key: str, model_name: str = "random_forest") -> Dict[str, Any]:
    path = artifact_path(dataset_key, model_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. Run training first: python -m src.train"
        )
    return joblib.load(path)


def list_available_models() -> List[Dict[str, str]]:
    out = []
    for p in sorted(MODELS_DIR.glob("*.joblib")):
        if p.name.endswith(".joblib") and "__" in p.stem:
            ds, model = p.stem.split("__", 1)
            out.append({"dataset_key": ds, "model_name": model, "path": str(p)})
    return out


def _row_from_mapping(mapping: Dict[str, Any], feature_names: List[str]) -> pd.DataFrame:
    missing = [f for f in feature_names if f not in mapping]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} required features. First missing: {missing[:5]}"
        )
    row = {f: float(mapping[f]) for f in feature_names}
    return pd.DataFrame([row], columns=feature_names)


def predict_from_features(
    dataset_key: str,
    features: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    model_name: str = "random_forest",
    top_k: int = 10,
) -> Dict[str, Any]:
    pack = load_model(dataset_key, model_name)
    feature_names = pack["feature_names"]
    pipe = pack["pipeline"]

    if isinstance(features, pd.DataFrame):
        if len(features) != 1:
            raise ValueError("Provide exactly one sample row in the DataFrame")
        X = features[feature_names]
        series = X.iloc[0]
    elif isinstance(features, pd.Series):
        X = _row_from_mapping(features.to_dict(), feature_names)
        series = X.iloc[0]
    else:
        X = _row_from_mapping(dict(features), feature_names)
        series = X.iloc[0]

    proba = float(pipe.predict_proba(X)[0, 1])
    pred = int(pipe.predict(X)[0])
    explanation = explain_prediction(pack, series, proba, pred, top_k=top_k)

    return {
        "dataset_key": dataset_key,
        "model_name": model_name,
        "language": pack.get("language"),
        "display_name": pack.get("display_name"),
        "prediction": explanation["decision"],
        "pred_label": pred,
        "probability_pd": proba,
        "probability_hc": 1.0 - proba,
        "explanation": explanation,
        "metrics_at_train_time": pack.get("metrics"),
        "disclaimer": pack.get("disclaimer"),
    }


def load_holdout_samples(dataset_key: str) -> pd.DataFrame:
    path = MODELS_DIR / f"{dataset_key}__holdout_samples.csv"
    if not path.exists():
        raise FileNotFoundError(f"Holdout samples not found: {path}. Train models first.")
    return pd.read_csv(path)
