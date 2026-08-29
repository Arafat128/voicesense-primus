"""Screening integrity stamp for Primus extraData.

Public stamp: model bundle hash + quality_ok + runtime.
It does NOT include P(PD), explanations, or audio. That is intentional so a
zkTLS attestation cannot leak a health label.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

from .privacy import utc_now_iso


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def make_integrity_stamp(
    result: Dict[str, Any],
    *,
    provenance_bundle_sha256: str,
    local_runtime: bool = True,
) -> Dict[str, Any]:
    quality = result.get("quality") or {}
    public_body = {
        "schema": "voicesense.integrity.v1",
        "model_bundle_sha256": provenance_bundle_sha256,
        "quality_ok": bool(quality.get("ok")),
        "runtime": "local" if local_runtime else "hosted",
        "status": result.get("status"),
        "language_mode": result.get("language_mode"),
        "audio_retained": False,
    }
    canonical = json.dumps(public_body, sort_keys=True, separators=(",", ":"))
    public_hash = _sha256_text(canonical)

    primary = result.get("primary") or {}
    private_body = {
        "decision_code": primary.get("decision_code"),
        "pred_label": primary.get("pred_label"),
        "generated_at": utc_now_iso(),
    }
    return {
        "public": public_body,
        "public_hash": public_hash,
        "private": private_body,
        "primus_addition_params": {
            "voicesense": "integrity-bind",
            "no_audio": True,
            "integrity_hash": public_hash,
            "model_bundle_sha256": provenance_bundle_sha256,
        },
    }


def stamp_from_session(stamp: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not stamp:
        return None
    return {
        "public_hash": stamp.get("public_hash"),
        "public": stamp.get("public"),
        "bound": bool(stamp.get("bound")),
    }
