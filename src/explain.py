"""Detailed PD / non-PD explanations for a single prediction."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import feature_meaning, DISCLAIMER


def explain_prediction(
    model_pack: Dict[str, Any],
    feature_row: pd.Series,
    proba_pd: float,
    pred_label: int,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Build a detailed explanation using:
    - model probability
    - RF global feature importance ranking
    - distance of each value to PD vs HC training class means
    - clinical glossary text

    A feature "supports PD" if the sample value is closer to the PD training mean
    than to the HC training mean (for that feature). This is clearer than generic
    high/low heuristics when class means are available.
    """
    feature_names: List[str] = model_pack["feature_names"]
    importances = model_pack.get("feature_importances") or []
    means = model_pack.get("train_means") or {}
    stds = model_pack.get("train_stds") or {}
    class_means = model_pack.get("class_means") or {}
    pd_means = class_means.get("PD") or {}
    hc_means = class_means.get("HC") or {}

    imp_map = {item["feature"]: float(item["importance"]) for item in importances}

    local_scores = []
    for feat in feature_names:
        val = float(feature_row[feat])
        mu = float(means.get(feat, np.nan)) if means.get(feat) is not None else np.nan
        sd = float(stds.get(feat, np.nan)) if stds.get(feat) is not None else np.nan
        z = 0.0 if (sd == 0 or np.isnan(sd)) else (val - mu) / sd
        importance = float(imp_map.get(feat, 0.0))

        mu_pd = pd_means.get(feat)
        mu_hc = hc_means.get(feat)
        supports_pd: Optional[bool] = None
        dist_pd = dist_hc = None
        if mu_pd is not None and mu_hc is not None:
            dist_pd = abs(val - float(mu_pd))
            dist_hc = abs(val - float(mu_hc))
            # closer to PD mean => supports PD; clear margin only
            if dist_pd < dist_hc:
                supports_pd = True
            elif dist_hc < dist_pd:
                supports_pd = False
            else:
                supports_pd = None

        # Local influence: importance * how class-discriminative this sample is
        class_sep = 0.0
        if dist_pd is not None and dist_hc is not None:
            class_sep = abs(dist_hc - dist_pd)
        local = (importance * (abs(z) + class_sep)) if importance else (abs(z) + class_sep)

        local_scores.append(
            {
                "feature": feat,
                "local_score": float(local),
                "value": val,
                "train_mean": None if np.isnan(mu) else float(mu),
                "pd_mean": None if mu_pd is None else float(mu_pd),
                "hc_mean": None if mu_hc is None else float(mu_hc),
                "z_score": float(z),
                "model_importance": importance,
                "supports_pd": supports_pd,
                "dist_to_pd_mean": None if dist_pd is None else float(dist_pd),
                "dist_to_hc_mean": None if dist_hc is None else float(dist_hc),
            }
        )

    local_scores.sort(key=lambda t: t["local_score"], reverse=True)
    top = local_scores[:top_k]

    why_pd: List[Dict[str, Any]] = []
    why_not: List[Dict[str, Any]] = []

    for item in top:
        feat = item["feature"]
        entry = {
            **item,
            "meaning": feature_meaning(feat),
            "comparison": _comparison_text(item),
            "supports": (
                "PD"
                if item["supports_pd"] is True
                else "Healthy / non-PD"
                if item["supports_pd"] is False
                else "neutral / unclear"
            ),
        }
        if item["supports_pd"] is True:
            why_pd.append(entry)
        elif item["supports_pd"] is False:
            why_not.append(entry)
        else:
            # neutral still shown under predicted side as influential but ambiguous
            if pred_label == 1:
                why_pd.append(entry)
            else:
                why_not.append(entry)

    decision = "PD" if pred_label == 1 else "Healthy Control (not PD)"
    confidence = proba_pd if pred_label == 1 else (1.0 - proba_pd)

    n_pd = sum(1 for t in top if t["supports_pd"] is True)
    n_hc = sum(1 for t in top if t["supports_pd"] is False)

    summary_lines = [
        f"Screening decision: {decision}",
        f"Model probability of PD: {proba_pd:.1%}",
        f"Confidence in decision: {confidence:.1%}",
        f"Dataset/model language: {model_pack.get('language', 'n/a')} ({model_pack.get('display_name', '')})",
        f"Classifier: {model_pack.get('model_name', 'n/a')}",
        (
            f"Among the top {top_k} influential features: "
            f"{n_pd} closer to PD training profile, {n_hc} closer to healthy training profile."
        ),
    ]

    if pred_label == 1:
        summary_lines.append(
            "Overall: the classifier scores this sample toward PD because its acoustic pattern "
            "is nearer the PD training profile on important dysphonia/spectral cues "
            "(e.g., pitch/loudness instability, noise, reduced harmonic quality, or MFCC shifts)."
        )
    else:
        summary_lines.append(
            "Overall: the classifier scores this sample toward healthy control because its acoustic "
            "pattern is nearer the healthy training profile on important cues "
            "(more stable phonation/loudness and/or healthier harmonic/spectral pattern)."
        )

    return {
        "decision": decision,
        "pred_label": int(pred_label),
        "probability_pd": float(proba_pd),
        "probability_hc": float(1.0 - proba_pd),
        "confidence": float(confidence),
        "summary_lines": summary_lines,
        "why_pd": why_pd,
        "why_not_pd": why_not,
        "top_influential_features": [
            {
                "feature": t["feature"],
                "local_score": t["local_score"],
                "value": t["value"],
                "z_score": t["z_score"],
                "importance": t["model_importance"],
                "supports": (
                    "PD"
                    if t["supports_pd"] is True
                    else "Healthy"
                    if t["supports_pd"] is False
                    else "neutral"
                ),
                "pd_mean": t["pd_mean"],
                "hc_mean": t["hc_mean"],
                "meaning": feature_meaning(t["feature"]),
            }
            for t in top
        ],
        "disclaimer": DISCLAIMER,
    }


def _comparison_text(item: Dict[str, Any]) -> str:
    feat = item["feature"]
    val = item["value"]
    parts = [f"Sample value = {val:.6g}."]
    if item.get("pd_mean") is not None and item.get("hc_mean") is not None:
        parts.append(
            f"Training PD mean = {item['pd_mean']:.6g}; "
            f"training healthy mean = {item['hc_mean']:.6g}."
        )
        if item["supports_pd"] is True:
            parts.append(
                f"This value is closer to the PD mean "
                f"(distance {item['dist_to_pd_mean']:.6g} vs healthy {item['dist_to_hc_mean']:.6g})."
            )
        elif item["supports_pd"] is False:
            parts.append(
                f"This value is closer to the healthy mean "
                f"(distance {item['dist_to_hc_mean']:.6g} vs PD {item['dist_to_pd_mean']:.6g})."
            )
        else:
            parts.append("Distances to PD and healthy means are essentially equal.")
    elif item.get("train_mean") is not None:
        rel = (
            "higher than"
            if val > item["train_mean"]
            else "lower than"
            if val < item["train_mean"]
            else "about equal to"
        )
        parts.append(f"It is {rel} the overall training mean ({item['train_mean']:.6g}).")
    parts.append(f"Model importance weight ≈ {item['model_importance']:.4f}; z-score ≈ {item['z_score']:.3f}.")
    return " ".join(parts)
