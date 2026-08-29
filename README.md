# VoiceSense — Parkinson’s Voice Screening

Premium research app for **voice-based Parkinson’s disease (PD) screening** with:

- **Live microphone / audio upload** (English & Bangla UI modes)
- **On-device processing** (local Streamlit: recording discarded after analysis)
- **Optional Primus zkTLS enrollment** (hashed uniqueness / age — never the WAV)
- **Feature-table lab** (UCI English + BenSParX Bengali)
- **Explainable outputs** (why PD / why not / uncertain)
- **Conservative decision policy** to reduce false PD flags on laptop mics

> **Not a medical diagnosis.** Screening support / research only.

## Quick start

```bash
git clone https://github.com/Arafat128/voicesense-primus.git
cd voicesense-primus
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py --server.port 8502
```

Open **http://localhost:8502**. This GitHub repo is the Primus-integrated copy, not the original thesis tree.

### Train models (if needed)

```bash
python -m src.train          # UCI + BenSParX feature models
python -m src.train_audio    # live acoustic model (Italian WAVs required under datasets/)
```

Datasets are expected under `../datasets/` relative to this project (see thesis folder layout), or adjust paths in `src/config.py` / `src/train_audio.py`.

## This copy (Primus Labs workspace)

This tree is a **copy** of the original thesis app. Work here so `D:\tmp\VoiceSense-PD` stays untouched.

Primus is wired in as an **identity / provenance / integrity** layer:

- zkTLS uniqueness / age / researcher / public GitHub templates (extension + Developer Hub)
- Local SHA-256 model provenance (`bundle_sha256`)
- Integrity stamp after Analyze (quality_ok + model bundle — **not** P(PD), **not** audio)
- Optional bind of that stamp into Primus `additionParams`
- Local attestation structure check + documented verifier addresses
- Optional Node sidecar `primus_sidecar/prove_github.cjs` for public GitHub zkTLS (core SDK)

Voice screening still does not call Primus.

## Privacy (on-device screening)

Local Streamlit is the supported privacy mode:

- Audio is scored in the Python process on **your machine**.
- A successful **Analyze voice** run **discards** the recording from Streamlit widget/session memory.
- **Discard recording** drops a clip without scoring.
- Export is an optional JSON **receipt with no WAV** and no upload filename.
- Decode may use a short-lived tempfile that is deleted immediately (librosa/container support).

**Streamlit Community Cloud is not on-device.** The browser must upload audio to the host. The app shows a warning in that case.

Enrollment receipts (fingerprints only) live in `.voicesense_local/` inside this copy, not in the original thesis app.

## Optional Primus enrollment

Primus is **not** on the microphone path. Use it only to prove a Web2 account or an age-style boolean, then store a SHA-256 fingerprint.

1. Create a project + template at [dev.primuslabs.xyz](https://dev.primuslabs.xyz/).
2. Copy `.env.example` to `.env` and fill `PRIMUS_APP_ID`, `PRIMUS_APP_SECRET` (local test signing only), and `PRIMUS_UNIQUENESS_TEMPLATE_ID`.
3. Build the browser SDK once:

```bash
cd enrollment
npm install
npm run build
```

4. Install the [Primus extension](https://chromewebstore.google.com/detail/primus/oeiomhmbaapihbilkfkhmlajkeegnjhe).
5. Open the **Privacy & enrollment** tab.

The SDK bundle is already built under `app/primus_enroll/frontend/vendor/` in this tree. Rebuild it after upgrading `@primuslabs/zktls-js-sdk`.

Without Hub credentials you can still paste an attestation JSON. `PRIMUS_MOCK=1` simulates enrollment for UI tests and is **not** a zkTLS proof.

Do not publish PD labels together with the enrollment fingerprint on a public chain.

## IDM popup (Error 0x80080005)

If **Internet Download Manager** shows *Cannot transfer the download to IDM*:

1. That is **not** an app crash — IDM intercepts browser media.
2. In the app, leave **Preview captured audio** **off** (default).
3. Or exclude `localhost` from IDM browser integration.

Analysis converts audio **in memory** to WAV and does not need a file download.

## Decision policy (live voice)

| Band | Rule (default) | Meaning |
|------|------------------|---------|
| Healthy / non-PD | P(PD) ≤ ~42% | No clear PD pattern |
| Uncertain | middle | Not enough evidence to flag PD |
| Possible PD | P(PD) ≥ ~72% | High score only — still not diagnosis |

## Project layout

```
app/streamlit_app.py   # premium UI
app/privacy_ui.py      # on-device privacy + Primus enrollment tab
app/primus_enroll/     # Streamlit iframe for zkTLS (identity only)
src/privacy.py         # receipts, audio wipe, local vs hosted
src/enrollment.py      # fingerprint-only enrollment records
enrollment/            # Vite bundle for @primuslabs/zktls-js-sdk
models/                # trained joblib artifacts
reports/               # metrics JSON
landing/               # static Vercel marketing page
```

## Deploy notes

| Target | Status |
|--------|--------|
| **Local Streamlit** | Full app (recommended) |
| **Streamlit Community Cloud** | Possible if system audio libs available |
| **Vercel** | Landing page only — ML stack is not serverless-friendly |

## Citation / data

Uses public corpora including UCI Parkinsons, BenSParX, and Italian Parkinson’s Voice (for live-audio training). Cite original dataset authors in academic work.

## License

Research / educational use. Provide proper dataset citations. Not for clinical deployment without regulatory validation.
