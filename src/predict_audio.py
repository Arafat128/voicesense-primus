"""
Predict PD from live/upload audio with quality gates and detailed explanation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd

from .audio_features import extract_features, vectorize
from .config import MODELS_DIR, DISCLAIMER
from .explain import explain_prediction
from .predict import load_model as load_feature_model


def _resolve_live_artifact(raw: Any, default_model: Optional[str] = None) -> Path:
    """
    Resolve a live-audio joblib path.

    Training used to store absolute paths from another machine
    (e.g. D:\\fugu\\Thesis\\pd_screening\\models\\...). Prefer the filename
    inside this project's models/ directory.
    """
    candidates = []
    if raw:
        p = Path(str(raw))
        # Always try this project's models/ first so CWD does not matter.
        candidates.append(MODELS_DIR / p.name)
        if p.is_absolute():
            candidates.append(p)
    if default_model:
        candidates.append(MODELS_DIR / f"live_audio_italian__{default_model}.joblib")
    for name in ("random_forest", "svm_rbf"):
        candidates.append(MODELS_DIR / f"live_audio_italian__{name}.joblib")

    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Live audio model not found under {MODELS_DIR}. "
        "Expected live_audio_italian__random_forest.joblib. Run: python -m src.train_audio"
    )


def load_live_audio_model(model_name: Optional[str] = None) -> Dict[str, Any]:
    pointer = MODELS_DIR / "live_audio_default.joblib"
    if model_name:
        path = _resolve_live_artifact(
            MODELS_DIR / f"live_audio_italian__{model_name}.joblib",
            default_model=model_name,
        )
        return joblib.load(path)
    if pointer.exists():
        p = joblib.load(pointer)
        path = _resolve_live_artifact(p.get("artifact"), default_model=p.get("default_model"))
        return joblib.load(path)
    path = _resolve_live_artifact(None)
    return joblib.load(path)


def _add_engineered(feat_dict: Dict[str, float]) -> Dict[str, float]:
    """Same engineered clinical features as training."""
    out = dict(feat_dict)
    eps = 1e-9
    f0m = out.get("praat_f0_mean")
    f0s = out.get("praat_f0_std")
    f0hi = out.get("praat_f0_max")
    f0lo = out.get("praat_f0_min")
    imx = out.get("praat_max_intensity")
    imn = out.get("praat_min_intensity")
    jit = out.get("praat_local_jitter")
    shi = out.get("praat_local_shimmer")
    if f0m is not None and f0s is not None and np.isfinite(f0m) and np.isfinite(f0s):
        out["f0_cv"] = float(f0s) / (abs(float(f0m)) + eps)
    if f0hi is not None and f0lo is not None and np.isfinite(f0hi) and np.isfinite(f0lo):
        out["f0_range"] = float(f0hi) - float(f0lo)
    if imx is not None and imn is not None and np.isfinite(imx) and np.isfinite(imn):
        out["intensity_range"] = float(imx) - float(imn)
    if jit is not None and shi is not None and np.isfinite(jit) and np.isfinite(shi):
        out["jitter_shimmer_product"] = float(jit) * float(shi)
    return out


def _decide_from_proba(proba: float, thresholds: Dict[str, float]) -> Dict[str, Any]:
    """
    Conservative decision for consumer mics:
    - high p => PD
    - low p => Healthy
    - middle => Uncertain (do NOT scare users with weak PD flags)
    """
    thr_pd = float(thresholds.get("pd_high", 0.72))
    thr_hc = float(thresholds.get("hc_high", 0.42))
    if proba >= thr_pd:
        return {
            "pred_label": 1,
            "decision_code": "PD",
            "prediction": "Possible PD pattern (screening only)",
            "band": "high_pd",
        }
    if proba <= thr_hc:
        return {
            "pred_label": 0,
            "decision_code": "HC",
            "prediction": "No clear PD pattern (likely non-PD on this sample)",
            "band": "high_hc",
        }
    return {
        "pred_label": -1,
        "decision_code": "UNCERTAIN",
        "prediction": "Uncertain — not enough evidence to flag PD",
        "band": "uncertain",
    }


def _predict_with_pack(pack: Dict[str, Any], feat_dict: Dict[str, float]) -> Dict[str, Any]:
    feat_dict = _add_engineered(feat_dict)
    names = pack["feature_names"]
    medians = pack.get("train_medians") or {}
    vec, imputed, completeness = vectorize(feat_dict, names, medians)
    X = pd.DataFrame([vec], columns=names)
    pipe = pack["pipeline"]
    proba = float(pipe.predict_proba(X)[0, 1])
    thresholds = pack.get("decision_thresholds") or {"pd_high": 0.72, "hc_high": 0.42}
    decision = _decide_from_proba(proba, thresholds)
    # For explanations, map uncertain toward the nearer class
    exp_label = 1 if proba >= 0.5 else 0
    series = X.iloc[0]
    explanation = explain_prediction(pack, series, proba, exp_label, top_k=12)
    # Override summary decision line for uncertain
    if decision["pred_label"] == -1:
        explanation["decision"] = decision["prediction"]
        explanation["summary_lines"] = [
            f"Screening decision: {decision['prediction']}",
            f"Model probability of PD: {proba:.1%}",
            (
                f"PD is only flagged when probability ≥ {thresholds.get('pd_high', 0.72):.0%}; "
                f"healthy when ≤ {thresholds.get('hc_high', 0.42):.0%}."
            ),
            "Your score is in the middle band — this often happens with short clips, new microphones, or normal voice variation.",
            "This is NOT a diagnosis. If you have concerns, consult a clinician.",
        ] + explanation.get("summary_lines", [])[3:]
    else:
        explanation["decision"] = decision["prediction"]
        explanation["summary_lines"][0] = f"Screening decision: {decision['prediction']}"

    return {
        "pred_label": decision["pred_label"],
        "decision_code": decision["decision_code"],
        "probability_pd": proba,
        "probability_hc": 1.0 - proba,
        "prediction": decision["prediction"],
        "decision_band": decision["band"],
        "thresholds": thresholds,
        "completeness": completeness,
        "imputed_features": imputed,
        "explanation": explanation,
        "model_name": pack.get("model_name"),
        "display_name": pack.get("display_name"),
        "language": pack.get("language"),
        "metrics_at_train_time": pack.get("metrics"),
        "false_positive_note": pack.get("false_positive_note"),
    }


def predict_from_audio(
    source: Union[str, Path, bytes, bytearray],
    language_mode: str = "english",
    live_model_name: Optional[str] = None,
    also_score_language_models: bool = True,
    require_quality_ok: bool = True,
) -> Dict[str, Any]:
    """
    language_mode: 'english' | 'bangla' | 'auto'
      - controls UI-facing model emphasis and optional UCI/BenSParX secondary scores
      - primary decision always uses the consistent live-audio model (same extractor)
    """
    extraction = extract_features(source)
    quality = extraction.quality.as_dict()

    if require_quality_ok and not extraction.quality.ok:
        return {
            "status": "rejected_quality",
            "quality": quality,
            "message": "Recording quality is not sufficient for reliable screening.",
            "messages": extraction.quality.messages,
            "disclaimer": DISCLAIMER,
        }

    live_pack = load_live_audio_model(live_model_name)
    primary = _predict_with_pack(live_pack, extraction.common)

    # Confidence blend: down-weight if quality borderline or many imputations
    q_factor = 1.0
    if extraction.quality.snr_db_proxy < 8:
        q_factor *= 0.9
    if extraction.quality.voiced_fraction < 0.25:
        q_factor *= 0.9
    if primary["completeness"] < 0.85:
        q_factor *= 0.85
    # For uncertain band, adjusted confidence reflects distance from decision edges
    if primary.get("pred_label") == -1:
        adjusted_conf = float(min(primary["probability_pd"], 1.0 - primary["probability_pd"]) * 2 * q_factor)
    elif primary.get("pred_label") == 1:
        adjusted_conf = float(primary["probability_pd"] * q_factor)
    else:
        adjusted_conf = float(primary["probability_hc"] * q_factor)

    secondary = {}
    warnings = [
        "Live laptop/phone mics differ from clinic recordings. "
        "A single clip can be wrong — do not treat a PD flag as a diagnosis.",
    ]
    if primary.get("pred_label") == 1:
        warnings.append(
            "PD pattern flagged only because probability was high. "
            "Re-record a quiet 5–8s sustained “aaa”/“আ—” and compare results."
        )
    # Secondary language models often over-flag PD on consumer mics (domain shift).
    # Keep them optional/supportive and apply the same conservative threshold wording.
    if also_score_language_models:
        try:
            uci_pack = load_feature_model("english_uci", "random_forest")
            uci_names = uci_pack["feature_names"]
            med = uci_pack.get("train_medians") or {}
            vec, imp, comp = vectorize(extraction.uci, uci_names, med)
            if comp >= 0.7:
                X = pd.DataFrame([vec], columns=uci_names)
                p = float(uci_pack["pipeline"].predict_proba(X)[0, 1])
                thr = live_pack.get("decision_thresholds") or {"pd_high": 0.72, "hc_high": 0.42}
                dec = _decide_from_proba(p, thr)
                secondary["english_uci"] = {
                    "prediction": dec["prediction"] + " [secondary]",
                    "probability_pd": p,
                    "completeness": comp,
                    "note": (
                        "Secondary English UCI-mapped score only. Often biased on laptop mics — "
                        "do not override the primary live model."
                    ),
                }
            else:
                warnings.append(f"UCI secondary skipped (feature completeness {comp:.0%}).")
        except Exception as e:
            warnings.append(f"UCI secondary unavailable: {e}")

        try:
            bn_pack = load_feature_model("bengali_bensparx", "random_forest")
            bn_names = bn_pack["feature_names"]
            med = bn_pack.get("train_medians") or {}
            vec, imp, comp = vectorize(extraction.bensparx, bn_names, med)
            if comp >= 0.25:
                X = pd.DataFrame([vec], columns=bn_names)
                p = float(bn_pack["pipeline"].predict_proba(X)[0, 1])
                thr = live_pack.get("decision_thresholds") or {"pd_high": 0.72, "hc_high": 0.42}
                dec = _decide_from_proba(p, thr)
                secondary["bengali_bensparx"] = {
                    "prediction": dec["prediction"] + " [secondary]",
                    "probability_pd": p,
                    "completeness": comp,
                    "imputed_count": len(imp),
                    "note": (
                        "Secondary Bengali BenSParX-mapped score only (partial features). "
                        "Supportive, not primary."
                    ),
                }
            else:
                warnings.append(f"BenSParX secondary skipped (completeness {comp:.0%}).")
        except Exception as e:
            warnings.append(f"BenSParX secondary unavailable: {e}")

    # Language-mode messaging
    if language_mode.lower() in ("english", "en"):
        mode_note = (
            "English mode: primary decision uses the live acoustic model (consistent train/serve features). "
            "UCI English secondary score is shown when mapping quality allows."
        )
        highlight = "english_uci"
    elif language_mode.lower() in ("bangla", "bengali", "bn"):
        mode_note = (
            "Bangla mode: primary decision uses the live acoustic model (acoustic PD cues are largely "
            "language-robust). BenSParX secondary score is shown when mapping quality allows."
        )
        highlight = "bengali_bensparx"
    else:
        mode_note = "Auto mode: primary live acoustic model; language secondaries shown if available."
        highlight = None

    return {
        "status": "ok",
        "language_mode": language_mode,
        "mode_note": mode_note,
        "quality": quality,
        "primary": {
            **primary,
            "adjusted_confidence": adjusted_conf,
            "role": "PRIMARY (same feature extractor as training — most trustworthy for live audio)",
        },
        "secondary": secondary,
        "highlight_secondary": highlight,
        "warnings": warnings,
        "extraction_preview": {
            "uci_subset": {k: extraction.uci[k] for k in list(extraction.uci)[:8]},
            "common_top": {
                item["feature"]: extraction.common.get(item["feature"])
                for item in (live_pack.get("feature_importances") or [])[:5]
                if isinstance(item, dict) and item.get("feature") in extraction.common
            },
        },
        "disclaimer": DISCLAIMER,
    }
