"""Load English UCI and Bengali BenSParX datasets with subject/speaker IDs."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .config import DATASETS


@dataclass
class DatasetBundle:
    key: str
    language: str
    display_name: str
    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    feature_names: List[str]
    sample_ids: np.ndarray
    positive_name: str
    negative_name: str


def _uci_subject_id(name: str) -> str:
    """phon_R01_S01_1 -> S01 (subject), not recording index."""
    parts = str(name).split("_")
    if len(parts) >= 3:
        return parts[2]
    return str(name)


def _bensparx_speaker_id(sample_id: str) -> str:
    """PD1_1 / HC12_3 -> PD1 / HC12 (speaker group)."""
    s = str(sample_id)
    if "_" in s:
        return s.rsplit("_", 1)[0]
    return s


def load_dataset(key: str) -> DatasetBundle:
    if key not in DATASETS:
        raise KeyError(f"Unknown dataset key: {key}. Choose from {list(DATASETS)}")

    meta = DATASETS[key]
    path = meta["path"]
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    label_col = meta["label_col"]
    id_col = meta["id_col"]

    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}' in {path}")
    if id_col not in df.columns:
        raise ValueError(f"Missing id column '{id_col}' in {path}")

    y = df[label_col].astype(int).to_numpy()
    sample_ids = df[id_col].astype(str).to_numpy()

    if key == "english_uci":
        groups = np.array([_uci_subject_id(x) for x in sample_ids])
        drop_cols = [id_col, label_col]
    elif key == "bengali_bensparx":
        groups = np.array([_bensparx_speaker_id(x) for x in sample_ids])
        drop_cols = [id_col, label_col]
    else:
        groups = sample_ids.copy()
        drop_cols = [id_col, label_col]

    X = df.drop(columns=drop_cols)
    # Keep only numeric feature columns
    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        # Median impute only for rare parse issues; report count
        na_count = int(X.isna().sum().sum())
        X = X.fillna(X.median(numeric_only=True))
        print(f"[warn] {key}: filled {na_count} missing numeric cells with column median")

    feature_names = list(X.columns)
    return DatasetBundle(
        key=key,
        language=meta["language"],
        display_name=meta["display_name"],
        X=X,
        y=y,
        groups=groups,
        feature_names=feature_names,
        sample_ids=sample_ids,
        positive_name=meta["positive_name"],
        negative_name=meta["negative_name"],
    )


def list_datasets() -> List[str]:
    return list(DATASETS.keys())
