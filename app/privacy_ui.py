"""Streamlit UI for on-device privacy + optional Primus enrollment."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

import streamlit as st

try:
    from app.notices import set_flash
except ImportError:
    from notices import set_flash  # type: ignore

import secrets

from src.enrollment import (
    attested_field_names,
    clear_enrollment,
    enrollment_unlocks_app,
    local_recipient,
    merge_enrollment,
    parse_attestation,
    primus_settings,
    public_enrollment_view,
    save_attestation_meta,
    save_enrollment,
    save_parse_debug,
    save_sid_unlock,
    load_sid_unlock,
    x_follow_profile_url,
)
from src.extension_bridge import (
    clear_extension_result,
    ensure_extension_server,
    read_extension_result,
    write_prove_config,
)
from src.integrity import stamp_from_session
from src.primus_onchain import local_attestation_checks
from src.privacy import is_local_runtime, make_screening_receipt, receipt_json
from src.provenance import collect_provenance, public_provenance_view


def _ensure_browser_sid() -> str:
    if "vs_sid" not in st.session_state:
        st.session_state["vs_sid"] = secrets.token_hex(16)
    return str(st.session_state["vs_sid"])


def browser_unlocked() -> bool:
    """True only for this browser's Streamlit session after a proof or owner PIN."""
    return st.session_state.get("live_unlocked") is True


def _session_enrollment() -> Optional[Dict[str, Any]]:
    """Per-browser Streamlit session only — never share unlock across browsers."""
    return st.session_state.get("enrollment")


def _commit_enrollment(record: Dict[str, Any]) -> Dict[str, Any]:
    merged = merge_enrollment(_session_enrollment(), record)
    merged["enrolled"] = enrollment_unlocks_app(merged)
    st.session_state["enrollment"] = merged
    st.session_state["live_unlocked"] = enrollment_unlocks_app(merged)
    # Disk copy is debug/receipt only. It must not unlock other browsers.
    sid = _ensure_browser_sid()
    save_enrollment({**merged, "session_sid": sid})
    if st.session_state["live_unlocked"]:
        save_sid_unlock(sid, merged)
    return merged


def _voicesense_public_origin() -> str:
    host = ""
    try:
        host = str(st.context.headers.get("Host") or "").strip()
    except Exception:
        host = ""
    if not host:
        host = "localhost:8502"
    return f"http://{host}"


def adopt_query_sid_unlock() -> None:
    """If this tab arrived with ?sid= after Primus, unlock it even if it is a new session."""
    if browser_unlocked():
        return
    qsid = str(st.query_params.get("sid") or "").strip()
    if not qsid:
        return
    rec = load_sid_unlock(qsid)
    if not rec or not enrollment_unlocks_app(rec):
        return
    st.session_state["vs_sid"] = qsid
    st.session_state["enrollment"] = rec
    st.session_state["live_unlocked"] = True


def render_privacy_banner() -> bool:
    local, _reason = is_local_runtime()
    if local:
        st.markdown(
            """
            <div class="vs-privacy ok">
              <strong>On-device processing.</strong>
              Your recording is scored on this computer and discarded after analysis.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="vs-privacy warn">
              <strong>This page is hosted.</strong>
              Audio is sent to the server that is running the app.
              Run VoiceSense on your own computer if the recording must stay local.
            </div>
            """,
            unsafe_allow_html=True,
        )
    return local


def render_export_receipt(result: Dict[str, Any], source_kind: Optional[str]) -> None:
    if not result or result.get("status") != "ok":
        return
    local, _ = is_local_runtime()
    st.markdown('<div class="vs-section">Save a copy (no recording)</div>', unsafe_allow_html=True)
    receipt = make_screening_receipt(
        result,
        source_kind=source_kind,
        include_features=False,
        enrollment=public_enrollment_view(_session_enrollment()),
        local_runtime=local,
        provenance=public_provenance_view(collect_provenance()),
        integrity=stamp_from_session(st.session_state.get("integrity_stamp")),
    )
    payload = receipt_json(receipt)
    st.download_button(
        "Download screening receipt (JSON, no recording)",
        data=payload.encode("utf-8"),
        file_name="voicesense-screening-receipt.json",
        mime="application/json",
        key="dl_receipt",
        use_container_width=True,
    )


def _apply_widget_result(value: Any, purpose: str, template_id: str) -> None:
    if not value:
        return
    marker = hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    seen_key = f"last_primus_payload_{purpose}"
    if st.session_state.get(seen_key) == marker:
        return
    st.session_state[seen_key] = marker
    if value.get("ok") is False and value.get("error"):
        set_flash("error", f"Verification failed: {value['error']}")
        st.rerun()
        return
    if not value.get("ok"):
        return
    att_preview = value.get("attestation")
    try:
        save_parse_debug(
            {
                "purpose": purpose,
                "field_names": value.get("data_keys")
                or attested_field_names(
                    att_preview.get("data") if isinstance(att_preview, dict) else None,
                    att_preview if isinstance(att_preview, dict) else None,
                ),
                "follow_extract": value.get("follow_extract"),
            }
        )
    except Exception:
        pass
    try:
        record = parse_attestation(
            value.get("attestation"),
            expected_template_id=template_id or value.get("template_id"),
            source="primus-zktls",
            sdk_verified=bool(value.get("sdk_verified")),
            purpose=purpose,
            extra_sources=value,
        )
        att_obj = value.get("attestation")
        fields = record.get("attested_field_names") or attested_field_names(
            (att_obj or {}).get("data") if isinstance(att_obj, dict) else None,
            att_obj if isinstance(att_obj, dict) else None,
        )
        save_parse_debug(
            {
                "purpose": purpose,
                "follow_ok": record.get("follow_ok"),
                "owner_ok": record.get("owner_ok"),
                "field_names": fields,
                "follow_extract": value.get("follow_extract"),
            }
        )
        if purpose == "x_follow" and record.get("follow_ok") is not True:
            shown = ", ".join(fields) if fields else "(none)"
            raise ValueError(
                "This proof does not show a follow of @its_perseus_1. "
                f"Attested fields: {shown}. "
                "In Primus Hub, add data.user.result.relationship_perspectives.following "
                "(or legacy.following) as a boolean named following."
            )
        if purpose == "x_owner" and record.get("owner_ok") is not True:
            raise ValueError(
                "That check is for the profile page, not your login. "
                "If you follow @its_perseus_1, use Prove follow."
            )
        saved = _commit_enrollment(record)
        checks = local_attestation_checks(value.get("attestation"))
        save_attestation_meta(
            {
                **checks,
                "purpose": purpose,
                "sdk_verified": True,
                "bound_integrity_hash": record.get("bound_integrity_hash"),
            }
        )
        if enrollment_unlocks_app(saved):
            set_flash("success", "Access granted. You can analyze your voice.")
        else:
            set_flash("error", "Verification did not unlock access. Follow @its_perseus_1 and prove that follow.")
        st.rerun()
    except Exception as exc:
        set_flash("error", f"Verification failed: {exc}")
        st.rerun()


def consume_pending_extension_proof() -> None:
    """Apply a proof written by the extension tab. Never displays secrets."""
    settings = primus_settings()
    pending = read_extension_result()
    if not pending or not pending.get("ok"):
        return
    mine = _ensure_browser_sid()
    proof_sid = str(pending.get("sid") or "")
    query_sid = str(st.query_params.get("sid") or "")
    if proof_sid and proof_sid not in (mine, query_sid):
        return
    if proof_sid and query_sid == proof_sid:
        st.session_state["vs_sid"] = proof_sid
    marker = hashlib.sha256(
        json.dumps(pending, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    if st.session_state.get("applied_extension_proof") == marker:
        return
    st.session_state["applied_extension_proof"] = marker
    purpose = pending.get("purpose") or "x_follow"
    tmpl = settings.get("x_follow_template_id") or ""
    clear_extension_result()
    _apply_widget_result(pending, purpose, tmpl or pending.get("template_id") or "")


def render_follow_gate(*, key_prefix: str = "live") -> bool:
    """
    User-facing follow + extension proof.
    Returns True when Analyze/record may proceed.
    App ID, secret, and template IDs are never shown.
    """
    consume_pending_extension_proof()
    adopt_query_sid_unlock()
    settings = primus_settings()
    handle = settings.get("x_follow_handle") or "its_perseus_1"
    follow_url = x_follow_profile_url()
    _ensure_browser_sid()
    if browser_unlocked():
        return True

    st.markdown(
        f"""
        <div class="vs-privacy warn">
          <strong>Follow @{handle} to continue.</strong>
          1. Open
          <a href="{follow_url}" target="_blank" rel="noopener">@{handle} on X</a>
          and follow.<br/>
          2. Come back and prove the follow with the Primus extension.
        </div>
        """,
        unsafe_allow_html=True,
    )

    prove_base = ensure_extension_server()
    write_prove_config(
        {
            "app_id": settings["app_id"],
            "app_secret": settings["app_secret"],
            "recipient": local_recipient(),
            "att_mode": settings["att_mode"],
            "addition_params": {},
            "sid": _ensure_browser_sid(),
            "return_url": (
                f"{_voicesense_public_origin()}/?verified=1&sid={_ensure_browser_sid()}"
            ),
            "flows": {
                "x_follow": {
                    "template_id": settings["x_follow_template_id"],
                    "field": settings["x_follow_field"],
                    "op": settings["x_follow_op"],
                    "value": settings["x_follow_value"],
                    "label": f"Prove you follow @{handle}",
                    "use_conditions": False,
                },
            },
        }
    )
    ready = bool(
        settings["app_id"] and settings["app_secret"] and settings["x_follow_template_id"]
    )
    st.link_button(
        f"Prove I follow @{handle}",
        f"{prove_base}?purpose=x_follow&sid={_ensure_browser_sid()}",
        disabled=not ready,
    )
    st.caption(
        "Opens a new tab. Stay logged into X and keep the Primus popup in front. "
        "When it succeeds you will return here automatically."
    )
    st.caption("Access is only for this browser tab. Another browser must prove follow separately.")

    @st.fragment(run_every=2)
    def _wait_for_extension_proof() -> None:
        if browser_unlocked():
            return
        pending = read_extension_result()
        if pending and pending.get("ok"):
            consume_pending_extension_proof()

    _wait_for_extension_proof()

    if settings.get("operator_pin"):
        from src.enrollment import operator_unlock_record, try_operator_unlock

        with st.expander("App owner"):
            pin = st.text_input("Unlock code", type="password", key=f"{key_prefix}_operator_pin")
            if st.button("Unlock", key=f"{key_prefix}_operator_go"):
                if try_operator_unlock(pin):
                    _commit_enrollment(operator_unlock_record())
                    set_flash("success", "Owner access granted.")
                    st.rerun()
                set_flash("error", "Unlock failed.")
                st.rerun()
    return False


def render_privacy_enrollment_tab() -> None:
    """Access status only — no credentials, no template IDs."""
    enrolled = _session_enrollment()
    if browser_unlocked():
        st.markdown(
            """
            <div class="vs-privacy ok">
              <strong>You're in.</strong>
              Recording and upload are unlocked on Live voice studio.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Sign out", type="secondary", key="clr_enroll"):
            clear_enrollment()
            st.session_state.pop("enrollment", None)
            st.session_state["live_unlocked"] = False
            set_flash("success", "Signed out on this browser.")
            st.rerun()
        return
    st.info("Follow and verify from Live voice studio to unlock recording.")
    render_follow_gate(key_prefix="access")
