"""
VoiceSense — Premium PD Voice Screening UI
Production-oriented Streamlit frontend with live audio + feature-table modes.
"""
from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from app.styles import PREMIUM_CSS
except ImportError:
    from styles import PREMIUM_CSS  # when launched as streamlit run app/streamlit_app.py

from src.config import DISCLAIMER, DATASETS, feature_meaning
from src.predict import (
    list_available_models,
    load_holdout_samples,
    load_model,
    predict_from_features,
)
from src.enrollment import local_recipient, primus_settings
from src.extension_bridge import ensure_extension_server, write_prove_config
from src.integrity import make_integrity_stamp
from src.privacy import bump_audio_nonce, is_local_runtime, wipe_audio_keys
from src.provenance import collect_provenance

try:
    from app.notices import render_flash, set_flash
    from app.privacy_ui import (
        browser_unlocked,
        consume_pending_extension_proof,
        render_export_receipt,
        render_follow_gate,
        render_privacy_banner,
        _session_enrollment,
    )
except ImportError:
    from notices import render_flash, set_flash  # type: ignore
    from privacy_ui import (  # type: ignore
        browser_unlocked,
        consume_pending_extension_proof,
        render_export_receipt,
        render_follow_gate,
        render_privacy_banner,
        _session_enrollment,
    )

st.set_page_config(
    page_title="VoiceSense · PD Screening",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

wipe_audio_keys(st.session_state)
if "audio_nonce" not in st.session_state:
    st.session_state["audio_nonce"] = 0

try:
    ensure_extension_server()
    _ps = primus_settings()
    write_prove_config(
        {
            "app_id": _ps["app_id"],
            "app_secret": _ps["app_secret"],
            "recipient": local_recipient(),
            "att_mode": _ps["att_mode"],
            "addition_params": {},
            "return_url": "http://127.0.0.1:8502/?verified=1",
            "flows": {
                "x_follow": {
                    "template_id": _ps["x_follow_template_id"],
                    "field": _ps["x_follow_field"],
                    "op": _ps["x_follow_op"],
                    "value": _ps["x_follow_value"],
                    "label": f"Prove you follow @{_ps.get('x_follow_handle') or 'its_perseus_1'}",
                    "use_conditions": False,
                },
            },
        }
    )
except Exception as _boot_exc:
    try:
        from src.enrollment import ENROLLMENT_DIR

        ENROLLMENT_DIR.mkdir(parents=True, exist_ok=True)
        (ENROLLMENT_DIR / "boot_error.txt").write_text(
            "".join(traceback.format_exception(type(_boot_exc), _boot_exc, _boot_exc.__traceback__)),
            encoding="utf-8",
        )
    except Exception:
        pass


def _live_audio_available() -> bool:
    if (ROOT / "models" / "live_audio_default.joblib").exists():
        return True
    return any((ROOT / "models").glob("live_audio_italian__*.joblib"))


def _to_wav_bytes(raw: bytes) -> bytes:
    """
    Normalize any browser/upload audio to WAV bytes for analysis.
    Avoids download-manager interception of odd media containers on analyze.
    """
    try:
        import librosa
        import soundfile as sf
        import numpy as np
        import tempfile

        # Detect container roughly
        suffix = ".wav"
        if raw[:4] == b"OggS":
            suffix = ".ogg"
        elif raw[:4] == b"\x1aE\xdf\xa3":
            suffix = ".webm"
        elif len(raw) > 12 and raw[4:8] == b"ftyp":
            suffix = ".mp4"
        elif raw[:3] == b"ID3" or raw[:2] == b"\xff\xfb":
            suffix = ".mp3"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            path = tmp.name
        try:
            y, sr = librosa.load(path, sr=22050, mono=True)
        finally:
            Path(path).unlink(missing_ok=True)

        # Peak normalize lightly for stable feature extraction
        peak = float(np.max(np.abs(y)) + 1e-9)
        y = (0.95 * y / peak).astype("float32")
        buf = io.BytesIO()
        sf.write(buf, y, sr, format="WAV")
        return buf.getvalue()
    except Exception:
        # Fall back to original bytes if conversion fails
        return raw


def _render_result_banner(primary: Dict[str, Any]) -> None:
    code = primary.get("decision_code") or (
        "PD" if primary.get("pred_label") == 1 else "HC" if primary.get("pred_label") == 0 else "UNCERTAIN"
    )
    css = "pd" if code == "PD" else "hc" if code == "HC" else "unc"
    st.markdown(
        f"""
        <div class="vs-result {css}">
          <div class="label">Primary screening decision</div>
          <div class="value">{primary.get('prediction', code)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_explanation(exp: dict, title: str = "Clinical-style explanation") -> None:
    st.markdown(f'<div class="vs-section">{title}</div>', unsafe_allow_html=True)
    with st.container():
        for line in exp.get("summary_lines", []):
            st.markdown(f"- {line}")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Signals toward PD")
            items = exp.get("why_pd") or []
            if not items:
                st.caption("No strong PD-favoring cues in the top features.")
            for item in items[:8]:
                with st.expander(f"{item['feature']} · score {item.get('local_score', 0):.3f}"):
                    st.write(item.get("comparison", ""))
                    st.caption(item.get("meaning", ""))
        with right:
            st.markdown("#### Signals toward healthy")
            items = exp.get("why_not_pd") or []
            if not items:
                st.caption("No strong healthy-favoring cues in the top features.")
            for item in items[:8]:
                with st.expander(f"{item['feature']} · score {item.get('local_score', 0):.3f}"):
                    st.write(item.get("comparison", ""))
                    st.caption(item.get("meaning", ""))

        top = exp.get("top_influential_features") or []
        if top:
            st.markdown("#### Top influential features")
            st.dataframe(pd.DataFrame(top), use_container_width=True, hide_index=True)


# ───────────────────────── Hero ─────────────────────────
st.markdown(
    """
    <div class="vs-hero">
      <div class="vs-kicker"><span></span> VoiceSense Research Platform</div>
      <h1 class="vs-title">Voice-based <em>Parkinson’s</em> screening</h1>
      <p class="vs-sub">
        Premium research interface for English &amp; Bangla voice analysis.
        Live microphone or feature-table input · explainable ML · screening support only.
      </p>
      <div class="vs-badges">
        <div class="vs-badge">On-device by default</div>
        <div class="vs-badge">Subject-aware models</div>
        <div class="vs-badge">Clinical acoustic features</div>
        <div class="vs-badge">Conservative PD threshold</div>
        <div class="vs-badge">Not a medical diagnosis</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(DISCLAIMER)
if st.query_params.get("verified") == "1":
    consume_pending_extension_proof()
    if browser_unlocked():
        try:
            st.query_params.clear()
        except Exception:
            pass
render_flash()
render_privacy_banner()
tab_live, tab_features, tab_about = st.tabs(
    ["Live voice studio", "Feature lab", "About"]
)

# ───────────────────────── LIVE ─────────────────────────
with tab_live:
    st.markdown('<div class="vs-section">Capture</div>', unsafe_allow_html=True)
    consume_pending_extension_proof()
    live_unlocked = browser_unlocked()
    if not live_unlocked:
        render_follow_gate(key_prefix="live")
    elif not _live_audio_available():
        st.error("Live audio model missing. Run: `python -m src.train_audio`")
    else:
        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.markdown(
                """
                <div class="vs-card">
                  <h3>Recording protocol</h3>
                  <p class="vs-muted">
                    1. Choose English or Bangla mode<br/>
                    2. Quiet room · mic 15–20 cm away<br/>
                    3. Sustain <b>“aaa” / “আ—”</b> for 5–8 seconds<br/>
                    4. Click Analyze — no diagnosis is made
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="vs-card">
                  <h3>Decision policy</h3>
                  <p class="vs-muted">
                    PD is flagged only when probability is <b>high (≥ ~72%)</b>.
                    Mid-range scores return <b>Uncertain</b> to reduce false alarms on laptop mics.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        lang = st.radio(
            "Language mode",
            options=["english", "bangla"],
            format_func=lambda x: "English" if x == "english" else "Bangla (বাংলা)",
            horizontal=True,
            key="lang_mode",
        )

        # IMPORTANT: optional preview only — auto audio widgets often trigger IDM popups
        preview = st.checkbox(
            "Preview captured audio in browser (optional — may open download managers like IDM)",
            value=False,
            key="preview_audio",
        )

        rec_col, up_col = st.columns(2)
        audio_raw: Optional[bytes] = None
        source_kind = None
        audio_nonce = int(st.session_state.get("audio_nonce", 0))

        with rec_col:
            st.markdown("##### Microphone")
            if hasattr(st, "audio_input"):
                mic = st.audio_input(
                    "Record a sustained vowel",
                    key=f"live_mic_{audio_nonce}",
                )
                if mic is not None:
                    audio_raw = mic.getvalue()
                    source_kind = "microphone"
                    st.success("Recording is in memory only — not written to disk.")
                    if preview:
                        st.audio(audio_raw)
            else:
                st.warning("Upgrade Streamlit for microphone support, or use file upload.")

        with up_col:
            st.markdown("##### Upload file")
            up = st.file_uploader(
                "WAV / MP3 / OGG / WEBM / M4A / FLAC",
                type=["wav", "mp3", "ogg", "webm", "m4a", "flac"],
                key=f"voice_up_{audio_nonce}",
            )
            if up is not None:
                audio_raw = up.getvalue()
                source_kind = "upload"
                st.success(
                    f"Loaded {len(audio_raw)//1024} KB in memory. "
                    "The filename is not kept in session state."
                )
                if preview:
                    st.audio(audio_raw)

        act_l, act_r = st.columns(2)
        with act_l:
            analyze = st.button(
                "Analyze voice",
                type="primary",
                use_container_width=True,
                disabled=audio_raw is None,
                key="analyze_live",
            )
        with act_r:
            discard = st.button(
                "Discard recording",
                type="secondary",
                use_container_width=True,
                disabled=audio_raw is None,
                key="discard_live",
            )
        if discard:
            bump_audio_nonce(st.session_state)
            st.session_state["audio_discarded_after_analysis"] = False
            set_flash("success", "Recording discarded. Nothing was scored or saved.")
            st.rerun()

        if analyze and audio_raw is not None:
            with st.spinner("Normalizing audio · extracting clinical features · scoring models…"):
                try:
                    from src.predict_audio import predict_from_audio

                    # Convert to WAV in-memory to stabilize decode path (no file download)
                    wav_bytes = _to_wav_bytes(audio_raw)
                    result = predict_from_audio(
                        wav_bytes,
                        language_mode=lang,
                        also_score_language_models=True,
                        require_quality_ok=True,
                    )
                    st.session_state["last_live_result"] = result
                    st.session_state["last_live_source"] = source_kind
                    st.session_state.pop("last_live_error", None)
                    st.session_state.pop("last_live_trace", None)
                    if result.get("status") == "ok":
                        primary = result.get("primary") or {}
                        decision = primary.get("prediction") or primary.get("decision_code") or "done"
                        local_rt, _ = is_local_runtime()
                        prov = collect_provenance()
                        st.session_state["integrity_stamp"] = make_integrity_stamp(
                            result,
                            provenance_bundle_sha256=prov["bundle_sha256"],
                            local_runtime=local_rt,
                        )
                        set_flash(
                            "success",
                            f"Analysis succeeded. Recording discarded from memory. Decision: {decision}",
                        )
                        bump_audio_nonce(st.session_state)
                        st.session_state["audio_discarded_after_analysis"] = True
                        st.rerun()
                    elif result.get("status") == "rejected_quality":
                        set_flash(
                            "error",
                            "Analysis failed: recording quality too low. Re-record a quiet 5–8s vowel.",
                        )
                        st.rerun()
                    else:
                        set_flash("error", f"Analysis failed: {result.get('status', 'unknown error')}")
                        st.rerun()
                except Exception as e:
                    set_flash("error", f"Analysis failed: {e}")
                    st.session_state["last_live_error"] = "".join(
                        traceback.format_exception_only(type(e), e)
                    )
                    st.session_state["last_live_trace"] = traceback.format_exc()
                    st.rerun()

        if st.session_state.get("audio_discarded_after_analysis"):
            st.caption("Recording discarded from memory after analysis.")

        if st.session_state.get("last_live_error"):
            st.error(st.session_state["last_live_error"])
            with st.expander("Technical traceback"):
                st.code(st.session_state.get("last_live_trace") or "")
            if st.button("Clear error", key="clr_live_err"):
                st.session_state.pop("last_live_error", None)
                st.session_state.pop("last_live_trace", None)
                st.rerun()

        result = st.session_state.get("last_live_result")
        if result:
            st.markdown('<div class="vs-section">Results</div>', unsafe_allow_html=True)
            st.caption(f"Source: {st.session_state.get('last_live_source', 'n/a')} · audio not retained")

            if result.get("status") == "rejected_quality":
                st.error(result.get("message", "Quality check failed"))
                for m in result.get("messages", []):
                    st.write(f"- {m}")
                st.json(result.get("quality", {}))
                st.info("Tip: longer steady vowel, quieter room, closer mic.")
            else:
                primary = result["primary"]
                _render_result_banner(primary)

                thr = primary.get("thresholds") or {}
                if thr:
                    st.caption(
                        f"Policy · PD if P≥{thr.get('pd_high', 0.72):.0%} · "
                        f"healthy if P≤{thr.get('hc_high', 0.42):.0%} · else Uncertain"
                    )

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("P(PD)", f"{primary['probability_pd']:.1%}")
                m2.metric("P(Healthy)", f"{primary['probability_hc']:.1%}")
                m3.metric("Adjusted confidence", f"{primary.get('adjusted_confidence', 0):.1%}")
                q = result.get("quality", {})
                m4.metric("Voiced fraction", f"{q.get('voiced_fraction', 0):.0%}")

                st.markdown(
                    f"**Quality** · duration {q.get('duration_sec', 0):.2f}s · "
                    f"SNR proxy {q.get('snr_db_proxy', 0):.1f} dB · OK={q.get('ok')}"
                )
                for msg in q.get("messages", []):
                    st.caption(msg)

                if primary.get("pred_label") == 1:
                    st.warning(
                        "Possible PD pattern on this clip only. Not a diagnosis. "
                        "Re-record a quiet 5–8s vowel and compare. See a clinician for real symptoms."
                    )
                elif primary.get("pred_label") == -1:
                    st.info(
                        "Uncertain is common on home mics and is safer than a false PD alarm. "
                        "Try one cleaner re-record if you want a clearer score."
                    )
                else:
                    st.success("No clear PD acoustic pattern on this sample.")

                st.markdown(
                    f"**Primary model:** {primary.get('display_name')} · `{primary.get('model_name')}` · "
                    f"completeness {primary.get('completeness', 0):.0%}"
                )
                if primary.get("metrics_at_train_time"):
                    m = primary["metrics_at_train_time"]
                    st.caption(
                        f"Train-time subject-aware: acc={m.get('accuracy', 0):.3f} · "
                        f"F1={m.get('f1', 0):.3f} · AUC={m.get('roc_auc')} · "
                        f"CV F1={m.get('cv_f1_mean', 0):.3f}±{m.get('cv_f1_std', 0):.3f}"
                    )

                _render_explanation(primary.get("explanation") or {}, "Why this decision")

                sec = result.get("secondary") or {}
                if sec:
                    st.markdown('<div class="vs-section">Secondary scores (supportive only)</div>', unsafe_allow_html=True)
                    for k, v in sec.items():
                        st.write(
                            f"**{k}**: {v['prediction']} · P(PD)={v['probability_pd']:.1%} · "
                            f"completeness={v.get('completeness', 0):.0%}"
                        )
                        st.caption(v.get("note", ""))

                for w in result.get("warnings") or []:
                    st.caption(f"⚠ {w}")

                st.warning(DISCLAIMER)
                render_export_receipt(result, st.session_state.get("last_live_source"))

# ───────────────────────── FEATURE LAB ─────────────────────────
with tab_features:
    st.markdown('<div class="vs-section">Feature-table screening</div>', unsafe_allow_html=True)
    consume_pending_extension_proof()
    feat_unlocked = browser_unlocked()
    if not feat_unlocked:
        render_follow_gate(key_prefix="feat")
    else:
        models = list_available_models()
        ds_keys = sorted(
            {
                m["dataset_key"]
                for m in models
                if m["dataset_key"] in DATASETS
            }
        )
        if not ds_keys:
            st.error("No feature models found. Run `python -m src.train`")
        else:
            dataset_labels = {k: DATASETS.get(k, {}).get("display_name", k) for k in ds_keys}
            dataset_key = st.selectbox(
                "Dataset model",
                options=ds_keys,
                format_func=lambda k: dataset_labels.get(k, k),
                key="feat_ds",
            )
            model_options = sorted(
                {m["model_name"] for m in models if m["dataset_key"] == dataset_key}
            )
            default_ix = model_options.index("random_forest") if "random_forest" in model_options else 0
            model_name = st.selectbox("Classifier", model_options, index=default_ix, key="feat_model")
            pack = load_model(dataset_key, model_name)
            feature_names = pack["feature_names"]
            st.caption(
                f"{pack.get('display_name')} · {len(feature_names)} features · {pack.get('language')}"
            )

            t1, t2, t3 = st.tabs(["Demo hold-out", "CSV upload", "Manual"])
            sample_features = None
            sample_meta: Dict[str, Any] = {}

            with t1:
                try:
                    holdout = load_holdout_samples(dataset_key)
                    options = [
                        f"{r.sample_id} | {r.subject_id} | true={'PD' if int(r.true_label)==1 else 'HC'}"
                        for r in holdout.itertuples()
                    ]
                    choice = st.selectbox("Sample", options, key="hold_sel")
                    row = holdout.iloc[options.index(choice)]
                    sample_meta = {
                        "sample_id": row["sample_id"],
                        "true_label": int(row["true_label"]),
                    }
                    sample_features = row[feature_names]
                    st.dataframe(pd.DataFrame(sample_features).T, use_container_width=True, hide_index=True)
                except FileNotFoundError as e:
                    st.error(str(e))

            with t2:
                up = st.file_uploader("CSV", type=["csv"], key="csv_feat")
                if up is not None:
                    df_up = pd.read_csv(up)
                    st.dataframe(df_up.head(), use_container_width=True)
                    row_ix = st.number_input("Row", 0, max(0, len(df_up) - 1), 0, key="csv_row")
                    missing = [c for c in feature_names if c not in df_up.columns]
                    if missing:
                        st.error(f"Missing {len(missing)} columns e.g. {missing[:5]}")
                    else:
                        sample_features = df_up.iloc[int(row_ix)][feature_names]
                        sample_meta = {"source": "csv"}

            with t3:
                imp = pack.get("feature_importances") or []
                top_feats = [x["feature"] for x in imp[:10]] if imp else feature_names[:10]
                medians = pack.get("train_medians") or {}
                cols = st.columns(2)
                manual_vals = {}
                for i, feat in enumerate(top_feats):
                    with cols[i % 2]:
                        manual_vals[feat] = st.number_input(
                            feat,
                            value=float(medians.get(feat, 0.0)),
                            format="%.6f",
                            help=feature_meaning(feat),
                            key=f"man_{feat}",
                        )
                if st.checkbox("Fill remaining with training medians", key="man_ok"):
                    full = {f: float(medians.get(f, 0.0)) for f in feature_names}
                    full.update(manual_vals)
                    sample_features = pd.Series(full)
                    sample_meta = {"source": "manual"}

            if st.button("Run feature screening", type="primary", key="run_feat"):
                if sample_features is None:
                    set_flash("error", "Feature screening failed: select or enter a sample first.")
                    st.rerun()
                else:
                    try:
                        out = predict_from_features(dataset_key, sample_features, model_name, top_k=12)
                        st.session_state["last_feat_result"] = out
                        st.session_state["last_feat_meta"] = sample_meta
                        set_flash(
                            "success",
                            f"Feature screening succeeded: {out.get('prediction', 'done')}",
                        )
                        st.rerun()
                    except Exception as e:
                        set_flash("error", f"Feature screening failed: {e}")
                        st.rerun()

            out = st.session_state.get("last_feat_result")
            if out:
                if out["pred_label"] == 1:
                    st.error(f"### {out['prediction']}")
                else:
                    st.success(f"### {out['prediction']}")
                a, b = st.columns(2)
                a.metric("P(PD)", f"{out['probability_pd']:.1%}")
                b.metric("P(Healthy)", f"{out['probability_hc']:.1%}")
                meta = st.session_state.get("last_feat_meta") or {}
                if meta.get("true_label") is not None:
                    true = "PD" if meta["true_label"] == 1 else "HC"
                    match = meta["true_label"] == out["pred_label"]
                    st.write(f"Hold-out true **{true}** · match **{'Yes' if match else 'No'}**")
                _render_explanation(out["explanation"])
                st.warning(DISCLAIMER)

# ───────────────────────── ABOUT ─────────────────────────
with tab_about:
    st.markdown(
        """
        <div class="vs-card">
          <h3>Method highlights</h3>
          <p class="vs-muted">
            English UCI &amp; Bengali BenSParX feature models · Live acoustic model trained with the
            same Praat/librosa extractor used at inference · subject/speaker-aware validation ·
            conservative thresholds for consumer microphones.
          </p>
          <h3>About the IDM popup</h3>
          <p class="vs-muted">
            Internet Download Manager sometimes intercepts browser media streams (Error 0x80080005).
            That is <b>not</b> a model crash. Leave audio preview off, or exclude localhost from IDM.
            Analysis uses in-memory WAV conversion and does not require a file download.
          </p>
          <h3>Privacy</h3>
          <p class="vs-muted">
            Live screening runs on this computer. The recording is discarded after analysis.
            Access is limited to people who follow the project account on X.
          </p>
          <h3>Disclaimer</h3>
          <p class="vs-muted">Research screening support only. Not a clinical diagnostic device.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="vs-footer">
      VoiceSense · BAUET thesis research prototype · Explainable voice screening · Not for clinical diagnosis
    </div>
    """,
    unsafe_allow_html=True,
)
