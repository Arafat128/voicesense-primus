"""Local model provenance — SHA-256 of screening artifacts.

This is the DVC-friendly half of Primus: hashes you can later bind into a
zkTLS extraData field. It never reads audio. It does not call Primus itself.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List

from .config import MODELS_DIR, REPORTS_DIR, ROOT

LIVE_MODELS = (
    "live_audio_italian__random_forest.joblib",
    "live_audio_italian__svm_rbf.joblib",
    "live_audio_default.joblib",
)
FEATURE_MODELS = (
    "english_uci__random_forest.joblib",
    "english_uci__svm_rbf.joblib",
    "bengali_bensparx__random_forest.joblib",
    "bengali_bensparx__svm_rbf.joblib",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _entry(rel: str, path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": rel, "present": False, "sha256": None, "bytes": 0}
    return {
        "path": rel,
        "present": True,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def collect_provenance() -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []
    for name in LIVE_MODELS + FEATURE_MODELS:
        files.append(_entry(f"models/{name}", MODELS_DIR / name))
    metrics = REPORTS_DIR / "live_audio_metrics.json"
    files.append(_entry("reports/live_audio_metrics.json", metrics))

    present = [f for f in files if f["present"] and f["sha256"]]
    bundle = hashlib.sha256(
        "|".join(f["sha256"] for f in present).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "voicesense.provenance.v1",
        "project_root": str(ROOT),
        "bundle_sha256": bundle,
        "files": files,
        "note": (
            "Local SHA-256 of model files. Bind bundle_sha256 into Primus "
            "additionParams — do not upload the joblib or any WAV."
        ),
    }


def public_provenance_view(full: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "bundle_sha256": full.get("bundle_sha256"),
        "files": [
            {"path": f["path"], "present": f["present"], "sha256": f.get("sha256")}
            for f in (full.get("files") or [])
        ],
    }
