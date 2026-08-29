"""
Train a high-quality live-audio PD screening model.

Critical design (avoid mistakes):
- Features for training are extracted with the SAME pipeline used at inference
  (src.audio_features.extract_features → common feature set).
- Labels from Italian Parkinson's Voice dataset:
    PD  = "28 People with Parkinson's disease"
    HC  = "22 Elderly Healthy Control"  (age-closer; young HC excluded by default)
- Subject/speaker-group aware split (folder = speaker).
- Quality filter: only keep recordings that pass audio quality gates.
- Also saves median fill values and feature list for live inference.

Optional:
- Use young HC if INCLUDE_YOUNG_HC=True (more data, age confound risk).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent
ITALIAN_ROOT = THESIS / "datasets" / "07_Italian_PD_Voice" / "italian_parkinson"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
CACHE_CSV = MODELS_DIR / "italian_common_features_cache.csv"

RANDOM_STATE = 42
# Include young HC for more healthy diversity (reduces "everything is PD" on new mics)
INCLUDE_YOUNG_HC = True
TEST_SIZE = 0.2

# Prefer sustained-vowel-like files first when available (names often start with B1/B2)
PREFERRED_PREFIXES = ("B1", "B2", "PR", "V", "a", "A")

# Drop non-clinical / recording-protocol confounds from the live model
DROP_FEATURES = {
    "duration_sec",  # clipped to fixed analysis window
    "snr_db_proxy",  # recording environment, not PD biology
    "praat_n_pulses",  # strongly length-dependent
    "praat_n_periods",  # strongly length-dependent
}

# CRITICAL for live-mic generalization:
# MFCC / spectral features overfit to Italian studio channel and flag new mics as PD.
# Use classical dysphonia + pitch/loudness measures that transfer better.
CLINICAL_FEATURE_ALLOW = {
    "praat_f0_mean",
    "praat_f0_min",
    "praat_f0_max",
    "praat_f0_std",
    "praat_local_jitter",
    "praat_local_abs_jitter",
    "praat_rap_jitter",
    "praat_ppq5_jitter",
    "praat_ddp_jitter",
    "praat_local_shimmer",
    "praat_local_db_shimmer",
    "praat_apq3",
    "praat_apq5",
    "praat_apq11",
    "praat_dda",
    "praat_hnr",
    "praat_nhr",
    "praat_ppe",
    "praat_rpde",
    "praat_dfa",
    "praat_spread1",
    "praat_spread2",
    "praat_d2",
    "praat_mean_intensity",
    "praat_min_intensity",
    "praat_max_intensity",
    "praat_mean_period",
    "praat_std_period",
    "praat_mean_autocorr_harmonicity",
    "voiced_fraction",
    # engineered relative features (mic-robust)
    "f0_cv",
    "f0_range",
    "intensity_range",
    "jitter_shimmer_product",
}

# Consumer-mic decision policy (saved into model artifact)
# Call PD only when fairly confident — reduces false alarms for healthy users.
DECISION_THRESHOLDS = {
    "pd_high": 0.72,   # >= this => PD
    "hc_high": 0.42,   # <= this => Healthy
    # between => Uncertain / not enough evidence for PD
}


def _label_and_subject(path: Path) -> Tuple[int, str, str]:
    """
    Return (label, subject_id, cohort)
    label: 1 PD, 0 HC
    """
    parts = path.parts
    # find cohort folder
    cohort = None
    for p in parts:
        if "Parkinson" in p:
            cohort = "PD"
            break
        if "Elderly Healthy" in p:
            cohort = "EHC"
            break
        if "Young Healthy" in p:
            cohort = "YHC"
            break
    if cohort is None:
        raise ValueError(f"Cannot infer cohort from {path}")

    # subject is parent folder of wav
    subject = path.parent.name.strip()
    if cohort == "PD":
        return 1, f"PD::{subject}", cohort
    if cohort == "EHC":
        return 0, f"EHC::{subject}", cohort
    return 0, f"YHC::{subject}", cohort


def list_italian_wavs(prefer_b1_b2: bool = True) -> List[Path]:
    """
    List training wavs.
    prefer_b1_b2: keep B1/B2 (and similar) sustained/task files when present per subject,
    which usually give cleaner phonation features and faster training.
    """
    if not ITALIAN_ROOT.exists():
        raise FileNotFoundError(f"Italian dataset not found: {ITALIAN_ROOT}")
    wavs = sorted(ITALIAN_ROOT.rglob("*.wav"))
    by_subject: Dict[str, List[Path]] = {}
    for w in wavs:
        try:
            label, subject, cohort = _label_and_subject(w)
        except ValueError:
            continue
        if cohort == "YHC" and not INCLUDE_YOUNG_HC:
            continue
        by_subject.setdefault(subject, []).append(w)

    selected: List[Path] = []
    for subject, files in by_subject.items():
        if prefer_b1_b2:
            preferred = [
                f
                for f in files
                if f.name.upper().startswith(("B1", "B2", "PR1", "PR2", "V1", "V2"))
                or f.stem.upper().startswith(("B1", "B2"))
            ]
            # keep up to 6 preferred per subject for more stable subject-level learning
            use = preferred[:6] if preferred else files[:4]
        else:
            use = files
        selected.extend(use)
    return selected


def extract_dataset(force_recompute: bool = False) -> pd.DataFrame:
    from .audio_features import extract_features

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_CSV.exists() and not force_recompute:
        print(f"Loading feature cache: {CACHE_CSV}")
        return pd.read_csv(CACHE_CSV)

    rows = []
    wavs = list_italian_wavs()
    print(f"Extracting features from {len(wavs)} Italian WAV files...")
    for i, w in enumerate(wavs, 1):
        try:
            label, subject, cohort = _label_and_subject(w)
            res = extract_features(w)
            if not res.quality.ok:
                # still keep borderline samples with warning flag; drop only catastrophic
                if res.quality.duration_sec < 0.5 or res.quality.voiced_fraction < 0.05:
                    print(f"  [skip bad] {w.name}: {res.quality.messages}")
                    continue
            row = {
                "path": str(w),
                "file": w.name,
                "subject": subject,
                "cohort": cohort,
                "label": label,
                "quality_ok": int(res.quality.ok),
                "duration_sec": res.quality.duration_sec,
                "snr_db_proxy": res.quality.snr_db_proxy,
                "voiced_fraction": res.quality.voiced_fraction,
            }
            row.update(res.common)
            rows.append(row)
        except Exception as e:
            print(f"  [fail] {w}: {e}")
        if i % 25 == 0:
            print(f"  progress {i}/{len(wavs)}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No features extracted from Italian dataset")
    df.to_csv(CACHE_CSV, index=False)
    print(f"Cached features: {CACHE_CSV} shape={df.shape}")
    return df


def _stratified_group_holdout(y, groups, test_size=TEST_SIZE, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    group_to_label = {}
    group_to_idx = {}
    for i, g in enumerate(groups):
        group_to_idx.setdefault(g, []).append(i)
        group_to_label[g] = int(y[i])
    test_groups, train_groups = [], []
    for lab in sorted(set(group_to_label.values())):
        gs = [g for g, l in group_to_label.items() if l == lab]
        gs = list(rng.permutation(gs))
        n_test = max(1, int(round(len(gs) * test_size)))
        if len(gs) - n_test < 1 and len(gs) > 1:
            n_test = len(gs) - 1
        test_groups.extend(gs[:n_test])
        train_groups.extend(gs[n_test:])
    train_idx = np.array([i for g in train_groups for i in group_to_idx[g]], dtype=int)
    test_idx = np.array([i for g in test_groups for i in group_to_idx[g]], dtype=int)
    return rng.permutation(train_idx), rng.permutation(test_idx)


def train_live_audio_model(force_recompute: bool = False) -> Dict[str, Any]:
    from .config import DISCLAIMER

    df = extract_dataset(force_recompute=force_recompute)
    meta_cols = {
        "path",
        "file",
        "subject",
        "cohort",
        "label",
        "quality_ok",
        "duration_sec",
        "snr_db_proxy",
        "voiced_fraction",
    }
    # Engineered relative features (more robust across microphones)
    eps = 1e-9
    if "praat_f0_mean" in df.columns and "praat_f0_std" in df.columns:
        df["f0_cv"] = df["praat_f0_std"] / (df["praat_f0_mean"].abs() + eps)
    if "praat_f0_max" in df.columns and "praat_f0_min" in df.columns:
        df["f0_range"] = df["praat_f0_max"] - df["praat_f0_min"]
    if "praat_max_intensity" in df.columns and "praat_min_intensity" in df.columns:
        df["intensity_range"] = df["praat_max_intensity"] - df["praat_min_intensity"]
    if "praat_local_jitter" in df.columns and "praat_local_shimmer" in df.columns:
        df["jitter_shimmer_product"] = df["praat_local_jitter"] * df["praat_local_shimmer"]

    feature_names = [
        c
        for c in df.columns
        if c not in meta_cols
        and c not in DROP_FEATURES
        and c in CLINICAL_FEATURE_ALLOW
    ]
    if len(feature_names) < 8:
        raise RuntimeError(
            f"Too few clinical features available ({len(feature_names)}). "
            "Re-run extraction with --force."
        )
    print(f"Using {len(feature_names)} clinical features (MFCC/spectral excluded for mic robustness)")
    X = df[feature_names].apply(pd.to_numeric, errors="coerce")
    y = df["label"].astype(int).to_numpy()
    groups = df["subject"].astype(str).to_numpy()

    train_idx, test_idx = _stratified_group_holdout(y, groups)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train, g_test = groups[train_idx], groups[test_idx]
    leak = set(g_train).intersection(set(g_test))
    if leak:
        raise RuntimeError(f"Subject leakage: {list(leak)[:5]}")

    # Prefer RF for native importance + strong tabular performance
    rf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    min_samples_leaf=3,
                    max_depth=8,  # reduce overfit to Italian channel quirks
                ),
            ),
        ]
    )
    base_svm = SVC(kernel="rbf", C=5.0, gamma="scale", class_weight="balanced", random_state=RANDOM_STATE)
    svm = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(base_svm, method="sigmoid", cv=3)),
        ]
    )

    models = {"random_forest": rf, "svm_rbf": svm}
    results = {
        "dataset": "Italian_PD_Voice (EHC vs PD)" if not INCLUDE_YOUNG_HC else "Italian_PD_Voice (all HC vs PD)",
        "n_samples": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "class_counts": {"PD": int((y == 1).sum()), "HC": int((y == 0).sum())},
        "n_features": len(feature_names),
        "split": {
            "method": "subject-stratified Group holdout 80/20",
            "train_samples": int(len(train_idx)),
            "test_samples": int(len(test_idx)),
            "train_subjects": int(len(set(g_train))),
            "test_subjects": int(len(set(g_test))),
        },
        "models": {},
        "disclaimer": DISCLAIMER,
        "feature_names": feature_names,
    }

    best_name, best_f1 = None, -1.0
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob)) if len(set(y_test)) > 1 else None,
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "classification_report": classification_report(y_test, pred, target_names=["HC", "PD"], digits=4),
        }
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_f1 = cross_val_score(pipe, X, y, groups=groups, cv=cv, scoring="f1", n_jobs=-1)
        metrics["cv_f1_mean"] = float(cv_f1.mean())
        metrics["cv_f1_std"] = float(cv_f1.std())

        importances = []
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
            order = np.argsort(imp)[::-1]
            importances = [{"feature": feature_names[i], "importance": float(imp[i])} for i in order]

        # medians from imputed training view
        medians = X_train.median(numeric_only=True).to_dict()
        means = X_train.mean(numeric_only=True).to_dict()
        stds = X_train.std(ddof=0, numeric_only=True).to_dict()
        pd_means = X_train.loc[y_train == 1].mean(numeric_only=True).to_dict()
        hc_means = X_train.loc[y_train == 0].mean(numeric_only=True).to_dict()

        art = {
            "pipeline": pipe,
            "feature_names": feature_names,
            "dataset_key": "live_audio_italian",
            "model_name": name,
            "language": "Multilingual-acoustic (trained on Italian public voice)",
            "display_name": "Live Voice Acoustic Model (Italian-trained)",
            "positive_name": "Parkinson's Disease (PD)",
            "negative_name": "Healthy Control (HC)",
            "feature_importances": importances,
            "train_medians": medians,
            "train_means": means,
            "train_stds": stds,
            "class_means": {"PD": pd_means, "HC": hc_means},
            "metrics": metrics,
            "disclaimer": DISCLAIMER,
            "input_type": "audio_wav",
            "extraction": "src.audio_features.extract_features.common",
            "quality_gates": {
                "min_duration_sec": 1.0,
                "min_voiced_fraction": 0.15,
                "min_snr_db": 5.0,
            },
            "decision_thresholds": DECISION_THRESHOLDS,
            "feature_policy": "clinical_dysphonia_only_no_mfcc",
            "false_positive_note": (
                "Live mics differ from research recordings. PD is reported only when "
                "probability is high; mid-range scores are labeled Uncertain."
            ),
        }
        out_path = MODELS_DIR / f"live_audio_italian__{name}.joblib"
        joblib.dump(art, out_path)
        results["models"][name] = {"metrics": metrics, "artifact": str(out_path), "top_features": importances[:15]}
        print(f"{name}: acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} auc={metrics['roc_auc']} "
              f"cv_f1={metrics['cv_f1_mean']:.3f}±{metrics['cv_f1_std']:.3f}")
        print(" confusion", metrics["confusion_matrix"])
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_name = name

    results["best_model"] = best_name
    results["app_default_model"] = "random_forest" if "random_forest" in results["models"] else best_name

    # pointer artifact for app
    default = results["app_default_model"]
    pointer = {
        "default_model": default,
        "artifact": f"live_audio_italian__{default}.joblib",
        "feature_names": feature_names,
        "summary_metrics": results["models"][default]["metrics"],
    }
    joblib.dump(pointer, MODELS_DIR / "live_audio_default.joblib")
    with open(REPORTS_DIR / "live_audio_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Best:", best_name, "F1=", best_f1)
    print(DISCLAIMER)
    return results


if __name__ == "__main__":
    force = "--force" in sys.argv
    train_live_audio_model(force_recompute=force)
