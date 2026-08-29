"""Streamlit component wrapper for optional Primus zkTLS enrollment."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import streamlit.components.v1 as components

_FRONTEND = Path(__file__).resolve().parent / "frontend"
_INDEX = _FRONTEND / "index.html"
_BUNDLE = _FRONTEND / "vendor" / "zktls-bundle.js"

_component = None
if _INDEX.exists():
    _component = components.declare_component(
        "voicesense_primus_enroll",
        path=str(_FRONTEND),
    )


def available() -> bool:
    return _component is not None


def sdk_bundle_present() -> bool:
    if not _BUNDLE.exists():
        return False
    try:
        text = _BUNDLE.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "PrimusZKTLS" in text and len(text) > 2000


def primus_enroll(
    *,
    app_id: str,
    template_id: str,
    recipient: str,
    app_secret: str = "",
    att_field: str = "screen_name",
    att_op: str = "SHA256",
    att_value: str = "",
    button_label: str = "Prove with Primus",
    purpose: str = "uniqueness",
    att_mode: str = "proxytls",
    addition_params: str = "",
    key: Optional[str] = None,
) -> Any:
    if _component is None:
        return None
    return _component(
        app_id=app_id,
        template_id=template_id,
        recipient=recipient,
        app_secret=app_secret,
        att_field=att_field,
        att_op=att_op,
        att_value=att_value,
        button_label=button_label,
        purpose=purpose,
        att_mode=att_mode or "proxytls",
        addition_params=addition_params or "",
        sdk_present=sdk_bundle_present(),
        key=key,
        default=None,
    )
