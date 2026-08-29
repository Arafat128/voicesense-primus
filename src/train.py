"""
Train PD screening models on English (UCI) and Bengali (BenSParX).

Correct methodology:
- Subject/speaker-group aware train/test split (no person leakage)
- StandardScaler fit on training fold only
- Random Forest + RBF-SVM
- Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
- Saves models, scalers, metrics, feature importances, example holdout samples
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from .config import MODELS_DIR, REPORTS_DIR, RANDOM_STATE, TEST_SIZE, DISCLAIMER
from .data_loading import load_dataset, list_datasets


def _metrics(y_true, y_pred, y_prob) -> Dict[str, Any]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["roc_auc"] = None
    return out


def _build_models() -> Dict[str, Pipeline]:
    rf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    criterion="gini",
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    # sklearn >=1.9 deprecates SVC(probability=True); calibrate for predict_proba
    base_svm = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    svm = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                CalibratedClassifierCV(base_svm, method="sigmoid", cv=3),
            ),
        ]
    )
    return {"random_forest": rf, "svm_rbf": svm}


def _stratified_group_holdout(
    y: np.ndarray, groups: np.ndarray, test_size: float = TEST_SIZE, seed: int = RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subject-aware hold-out that also balances PD/HC subjects in train and test.

    GroupShuffleSplit alone can put almost all healthy subjects on one side when
    healthy subject count is small (UCI has only 8 HC subjects). That creates a
    misleading hold-out. This routine samples test subjects within each class.
    """
    rng = np.random.RandomState(seed)
    # one label per group (datasets guarantee pure groups)
    group_to_label = {}
    group_to_indices = {}
    for i, g in enumerate(groups):
        group_to_indices.setdefault(g, []).append(i)
        group_to_label[g] = int(y[i])

    test_groups = []
    train_groups = []
    for label in sorted(set(group_to_label.values())):
        g_list = [g for g, lab in group_to_label.items() if lab == label]
        g_list = list(rng.permutation(g_list))
        n_test = max(1, int(round(len(g_list) * test_size)))
        # leave at least 1 train subject per class when possible
        if len(g_list) - n_test < 1 and len(g_list) > 1:
            n_test = len(g_list) - 1
        test_groups.extend(g_list[:n_test])
        train_groups.extend(g_list[n_test:])

    test_set = set(test_groups)
    train_idx = np.array(
        [i for g in train_groups for i in group_to_indices[g]], dtype=int
    )
    test_idx = np.array(
        [i for g in test_groups for i in group_to_indices[g]], dtype=int
    )
    # shuffle sample order
    train_idx = rng.permutation(train_idx)
    test_idx = rng.permutation(test_idx)
    return train_idx, test_idx


def _group_holdout_split(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """80/20 hold-out that keeps all samples of a subject on one side (class-stratified by subject)."""
    return _stratified_group_holdout(y, groups, test_size=TEST_SIZE, seed=RANDOM_STATE)


def _feature_importance(pipe: Pipeline, feature_names: List[str]) -> List[Dict[str, Any]]:
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
        order = np.argsort(imp)[::-1]
        return [
            {"feature": feature_names[i], "importance": float(imp[i])}
            for i in order
        ]
    # SVM: use absolute dual-less linear proxy is not available for RBF;
    # fall back to permutation-free empty and compute later via mean |z| separation.
    return []


def train_one(dataset_key: str) -> Dict[str, Any]:
    data = load_dataset(dataset_key)
    X, y, groups = data.X, data.y, data.groups
    feature_names = data.feature_names

    train_idx, test_idx = _group_holdout_split(X, y, groups)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train, g_test = groups[train_idx], groups[test_idx]

    # Leakage check
    leak = set(g_train).intersection(set(g_test))
    if leak:
        raise RuntimeError(f"Subject leakage detected for {dataset_key}: {sorted(leak)[:10]}")

    models = _build_models()
    results: Dict[str, Any] = {
        "dataset_key": dataset_key,
        "display_name": data.display_name,
        "language": data.language,
        "n_samples": int(len(y)),
        "n_features": len(feature_names),
        "n_subjects": int(len(set(groups))),
        "class_counts": {
            "PD": int((y == 1).sum()),
            "HC": int((y == 0).sum()),
        },
        "split": {
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "train_subjects": int(len(set(g_train))),
            "test_subjects": int(len(set(g_test))),
            "method": "GroupShuffleSplit 80/20 (subject/speaker aware)",
            "random_state": RANDOM_STATE,
        },
        "models": {},
        "disclaimer": DISCLAIMER,
        "feature_names": feature_names,
    }

    best_name = None
    best_f1 = -1.0
    best_pipe = None

    for model_name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        metrics = _metrics(y_test, y_pred, y_prob)

        # Subject-aware 5-fold CV on full data for stability estimate
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            pipe,
            X,
            y,
            groups=groups,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
        )
        metrics["cv_f1_mean"] = float(cv_scores.mean())
        metrics["cv_f1_std"] = float(cv_scores.std())
        metrics["classification_report"] = classification_report(
            y_test, y_pred, target_names=["HC", "PD"], digits=4
        )

        importances = _feature_importance(pipe, feature_names)
        # For SVM without native importance, use RF importances later; store empty.

        # Class-conditional training stats for faithful explanations
        pd_mask = y_train == 1
        hc_mask = y_train == 0
        class_means = {
            "PD": X_train.loc[pd_mask].mean().to_dict() if pd_mask.any() else {},
            "HC": X_train.loc[hc_mask].mean().to_dict() if hc_mask.any() else {},
        }
        class_stds = {
            "PD": X_train.loc[pd_mask].std(ddof=0).to_dict() if pd_mask.any() else {},
            "HC": X_train.loc[hc_mask].std(ddof=0).to_dict() if hc_mask.any() else {},
        }

        out_path = MODELS_DIR / f"{dataset_key}__{model_name}.joblib"
        joblib.dump(
            {
                "pipeline": pipe,
                "feature_names": feature_names,
                "dataset_key": dataset_key,
                "model_name": model_name,
                "language": data.language,
                "display_name": data.display_name,
                "positive_name": data.positive_name,
                "negative_name": data.negative_name,
                "feature_importances": importances,
                "train_medians": X_train.median().to_dict(),
                "train_means": X_train.mean().to_dict(),
                "train_stds": X_train.std(ddof=0).to_dict(),
                "class_means": class_means,
                "class_stds": class_stds,
                "metrics": metrics,
                "disclaimer": DISCLAIMER,
            },
            out_path,
        )

        results["models"][model_name] = {
            "metrics": metrics,
            "artifact": str(out_path),
            "top_features": importances[:15],
        }

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_name = model_name
            best_pipe = pipe

    results["best_model"] = best_name

    # If SVM has no importances, copy RF importances for explanation ranking baseline
    rf_imp = results["models"].get("random_forest", {}).get("top_features", [])
    if rf_imp:
        svm_art = MODELS_DIR / f"{dataset_key}__svm_rbf.joblib"
        if svm_art.exists():
            pack = joblib.load(svm_art)
            if not pack.get("feature_importances"):
                pack["feature_importances"] = joblib.load(
                    MODELS_DIR / f"{dataset_key}__random_forest.joblib"
                )["feature_importances"]
                joblib.dump(pack, svm_art)

    # Save holdout sample bank for the app (safe demo inputs from TEST set only)
    holdout = X_test.copy()
    holdout.insert(0, "sample_id", data.sample_ids[test_idx])
    holdout.insert(1, "subject_id", g_test)
    holdout.insert(2, "true_label", y_test)
    holdout_path = MODELS_DIR / f"{dataset_key}__holdout_samples.csv"
    holdout.to_csv(holdout_path, index=False)
    results["holdout_samples_path"] = str(holdout_path)

    # Prefer RF for default app explanations (native feature importance)
    default_model = "random_forest" if "random_forest" in results["models"] else best_name
    results["app_default_model"] = default_model

    report_path = REPORTS_DIR / f"{dataset_key}__metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== {data.display_name} ===")
    print(f"Subjects train/test: {results['split']['train_subjects']}/{results['split']['test_subjects']}")
    for mn, md in results["models"].items():
        m = md["metrics"]
        print(
            f"  {mn}: acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
            f"rec={m['recall']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']} "
            f"cv_f1={m['cv_f1_mean']:.4f}±{m['cv_f1_std']:.4f}"
        )
        print("  confusion [ [TN,FP],[FN,TP] ]:", m["confusion_matrix"])
    print(f"Best by holdout F1: {best_name}")
    print(f"Saved report: {report_path}")
    return results


def train_all() -> Dict[str, Any]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for key in list_datasets():
        summary[key] = train_one(key)
    summary_path = REPORTS_DIR / "training_summary.json"
    # store compact summary
    compact = {
        k: {
            "display_name": v["display_name"],
            "best_model": v["best_model"],
            "app_default_model": v["app_default_model"],
            "split": v["split"],
            "models": {
                mn: md["metrics"] for mn, md in v["models"].items()
            },
        }
        for k, v in summary.items()
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2)
    print(f"\nAll done. Summary: {summary_path}")
    print(DISCLAIMER)
    return summary


if __name__ == "__main__":
    train_all()
