"""Live success / fail notices that survive Streamlit reruns."""
from __future__ import annotations

from typing import Literal

import streamlit as st

Kind = Literal["success", "error", "warning", "info"]
FLASH_KEY = "vs_flash"


def set_flash(kind: Kind, text: str) -> None:
    st.session_state[FLASH_KEY] = {"kind": kind, "text": text}


def render_flash() -> None:
    flash = st.session_state.pop(FLASH_KEY, None)
    if not flash:
        return
    kind = flash.get("kind") or "info"
    text = str(flash.get("text") or "")
    if not text:
        return
    if kind == "success":
        st.success(text)
        st.toast(text, icon="✅")
    elif kind == "error":
        st.error(text)
        st.toast(text, icon="❌")
    elif kind == "warning":
        st.warning(text)
        st.toast(text, icon="⚠️")
    else:
        st.info(text)
        st.toast(text, icon="ℹ️")
