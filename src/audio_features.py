"""
High-quality acoustic feature extraction for live/upload PD voice screening.

Uses:
- Praat via parselmouth for clinical dysphonia measures (jitter, shimmer, HNR, pitch)
- librosa for MFCCs, energy, spectral descriptors

Produces:
1) common_audio feature vector (for the dedicated live-audio model trained on Italian WAVs)
2) UCI-like feature vector (for English UCI model when mapping is complete enough)
3) BenSParX-like feature vector (for Bengali model when mapping is complete enough)
4) quality report (duration, SNR proxy, voiced fraction) — reject bad takes
"""
from __future__ import annotations

import io
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

TARGET_SR = 22050
MIN_DURATION_SEC = 1.0
MIN_VOICED_FRACTION = 0.15
MIN_SNR_DB = 5.0


@dataclass
class AudioQuality:
    ok: bool
    duration_sec: float
    rms: float
    snr_db_proxy: float
    voiced_fraction: float
    messages: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "duration_sec": self.duration_sec,
            "rms": self.rms,
            "snr_db_proxy": self.snr_db_proxy,
            "voiced_fraction": self.voiced_fraction,
            "messages": self.messages,
        }


@dataclass
class ExtractionResult:
    quality: AudioQuality
    common: Dict[str, float]
    uci: Dict[str, float]
    bensparx: Dict[str, float]
    common_feature_names: List[str]
    uci_feature_names: List[str]
    bensparx_feature_names: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)


def _load_audio(source: Union[str, Path, bytes, bytearray], sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    import librosa
    import soundfile as sf

    if isinstance(source, (bytes, bytearray)):
        # Write temp file for robust decoding (webm/wav/ogg from browser)
        data = bytes(source)
        suffix = ".wav"
        # detect webm/ogg headers roughly
        if data[:4] == b"OggS":
            suffix = ".ogg"
        elif len(data) > 12 and data[4:8] == b"ftyp":
            suffix = ".mp4"
        elif data[:4] == b"\x1aE\xdf\xa3":
            suffix = ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            path = tmp.name
        try:
            y, file_sr = librosa.load(path, sr=sr, mono=True)
        finally:
            Path(path).unlink(missing_ok=True)
        return y.astype(np.float32), sr

    path = str(source)
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), sr


def _trim_and_normalize(y: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    import librosa

    if y.size == 0:
        return y
    y_trim, _ = librosa.effects.trim(y, top_db=top_db)
    if y_trim.size < int(0.3 * TARGET_SR):
        y_trim = y
    peak = np.max(np.abs(y_trim)) + 1e-9
    y_trim = 0.95 * y_trim / peak
    return y_trim


def _snr_proxy(y: np.ndarray) -> float:
    """Crude SNR: ratio of top-energy frames to bottom-energy frames."""
    if y.size < 100:
        return 0.0
    frame = 1024
    hop = 512
    energies = []
    for i in range(0, max(1, len(y) - frame), hop):
        energies.append(float(np.mean(y[i : i + frame] ** 2)))
    energies = np.array(energies) + 1e-12
    thr_hi = np.percentile(energies, 80)
    thr_lo = np.percentile(energies, 20)
    sig = energies[energies >= thr_hi].mean()
    noi = energies[energies <= thr_lo].mean()
    return float(10.0 * np.log10(sig / noi))


def _praat_voice_report(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Extract Praat-style dysphonia measures via parselmouth."""
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(y, sampling_frequency=sr)
    # Pitch
    pitch = call(snd, "To Pitch", 0.0, 75.0, 500.0)
    point_process = call(snd, "To PointProcess (periodic, cc)", 75.0, 500.0)

    def safe_call(*args, default=np.nan):
        try:
            v = call(*args)
            if v is None:
                return default
            v = float(v)
            if np.isnan(v) or np.isinf(v):
                return default
            return v
        except Exception:
            return default

    f0_mean = safe_call(pitch, "Get mean", 0, 0, "Hertz", default=np.nan)
    f0_min = safe_call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic", default=np.nan)
    f0_max = safe_call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic", default=np.nan)
    f0_std = safe_call(pitch, "Get standard deviation", 0, 0, "Hertz", default=np.nan)

    # Jitter
    local_jitter = safe_call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    local_abs_jitter = safe_call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rap_jitter = safe_call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5_jitter = safe_call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddp_jitter = safe_call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)

    # Shimmer
    local_shimmer = safe_call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    local_db_shimmer = safe_call([snd, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3 = safe_call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq5 = safe_call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11 = safe_call([snd, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    dda = safe_call([snd, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
    hnr = safe_call(harmonicity, "Get mean", 0, 0)
    # NHR approximation from HNR (dB): NHR ≈ 1 / (10^(HNR/10) + 1) style; use linear ratio proxy
    if hnr is not None and not np.isnan(hnr):
        # HNR in dB; convert rough noise ratio
        nhr = float(1.0 / (1.0 + 10.0 ** (hnr / 10.0)))
    else:
        nhr = np.nan

    # Pulses / periods
    n_pulses = safe_call(point_process, "Get number of points", default=0.0)
    n_periods = max(n_pulses - 1.0, 0.0) if not np.isnan(n_pulses) else 0.0
    mean_period = safe_call(point_process, "Get mean period", 0, 0, 0.0001, 0.02, 1.3)
    std_period = safe_call(point_process, "Get stdev period", 0, 0, 0.0001, 0.02, 1.3)

    # Intensity
    intensity = call(snd, "To Intensity", 75.0, 0.0, "yes")
    mean_int = safe_call(intensity, "Get mean", 0, 0, "energy")
    min_int = safe_call(intensity, "Get minimum", 0, 0, "Parabolic")
    max_int = safe_call(intensity, "Get maximum", 0, 0, "Parabolic")

    # Voiced frames from pitch
    n_frames = int(call(pitch, "Get number of frames"))
    voiced = 0
    f0_values = []
    for i in range(1, n_frames + 1):
        v = call(pitch, "Get value in frame", i, "Hertz")
        if v is not None and not np.isnan(float(v)) and float(v) > 0:
            voiced += 1
            f0_values.append(float(v))
    voiced_fraction = voiced / max(n_frames, 1)

    # Autocorrelation harmonicity proxy
    try:
        ac = float(call(snd, "Get absolute extremum", 0, 0, "None"))  # not ideal
    except Exception:
        ac = np.nan
    # Better: mean of harmonicity object already is HNR-like
    mean_autocorr_harm = hnr

    # PPE approximation from F0 contour (pitch period entropy proxy)
    ppe = _ppe_from_f0(np.array(f0_values, dtype=float))

    # RPDE / DFA light approximations from F0 residual
    rpde = _rpde_proxy(np.array(f0_values, dtype=float))
    dfa = _dfa_proxy(y)
    spread1, spread2 = _spread_proxies(np.array(f0_values, dtype=float))
    d2 = _d2_proxy(np.array(f0_values, dtype=float))

    return {
        "f0_mean": f0_mean,
        "f0_min": f0_min,
        "f0_max": f0_max,
        "f0_std": f0_std,
        "local_jitter": local_jitter,
        "local_abs_jitter": local_abs_jitter,
        "rap_jitter": rap_jitter,
        "ppq5_jitter": ppq5_jitter,
        "ddp_jitter": ddp_jitter,
        "local_shimmer": local_shimmer,
        "local_db_shimmer": local_db_shimmer,
        "apq3": apq3,
        "apq5": apq5,
        "apq11": apq11,
        "dda": dda,
        "hnr": hnr,
        "nhr": nhr,
        "n_pulses": float(n_pulses) if not np.isnan(n_pulses) else 0.0,
        "n_periods": float(n_periods),
        "mean_period": mean_period,
        "std_period": std_period,
        "mean_intensity": mean_int,
        "min_intensity": min_int,
        "max_intensity": max_int,
        "voiced_fraction": float(voiced_fraction),
        "mean_autocorr_harmonicity": mean_autocorr_harm if mean_autocorr_harm is not None else np.nan,
        "ppe": ppe,
        "rpde": rpde,
        "dfa": dfa,
        "spread1": spread1,
        "spread2": spread2,
        "d2": d2,
    }


def _ppe_from_f0(f0: np.ndarray) -> float:
    """Pitch period entropy proxy from log-F0 distribution (Tsanas-inspired)."""
    f0 = f0[np.isfinite(f0) & (f0 > 0)]
    if f0.size < 8:
        return np.nan
    logf0 = np.log(f0)
    # whitening residual via simple detrend
    x = logf0 - np.mean(logf0)
    hist, _ = np.histogram(x, bins=min(20, max(5, f0.size // 3)), density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return np.nan
    p = hist / hist.sum()
    ent = -np.sum(p * np.log(p + 1e-12))
    # normalize roughly to 0-1 range
    return float(ent / np.log(len(p) + 1e-12))


def _rpde_proxy(f0: np.ndarray) -> float:
    f0 = f0[np.isfinite(f0) & (f0 > 0)]
    if f0.size < 10:
        return np.nan
    d = np.diff(f0)
    if d.size < 5:
        return np.nan
    hist, _ = np.histogram(d, bins=15, density=True)
    hist = hist[hist > 0]
    p = hist / hist.sum()
    return float(-np.sum(p * np.log(p + 1e-12)) / np.log(len(p) + 1e-12))


def _dfa_proxy(y: np.ndarray, box: int = 64) -> float:
    """Very light DFA-like fluctuation slope proxy on amplitude envelope."""
    if y.size < box * 4:
        return np.nan
    env = np.abs(y)
    # integrated series
    y2 = np.cumsum(env - np.mean(env))
    sizes = [32, 64, 128, 256]
    sizes = [s for s in sizes if s * 4 < len(y2)]
    if len(sizes) < 2:
        return np.nan
    fs = []
    for s in sizes:
        nseg = len(y2) // s
        if nseg < 2:
            continue
        rms = []
        for i in range(nseg):
            seg = y2[i * s : (i + 1) * s]
            t = np.arange(s)
            coef = np.polyfit(t, seg, 1)
            trend = np.polyval(coef, t)
            rms.append(np.sqrt(np.mean((seg - trend) ** 2)) + 1e-12)
        fs.append(np.mean(rms))
    if len(fs) < 2:
        return np.nan
    slope = np.polyfit(np.log(sizes[: len(fs)]), np.log(fs), 1)[0]
    return float(slope)


def _spread_proxies(f0: np.ndarray) -> Tuple[float, float]:
    f0 = f0[np.isfinite(f0) & (f0 > 0)]
    if f0.size < 5:
        return np.nan, np.nan
    logf0 = np.log(f0)
    # spread1 ~ measure of variation (negative tendency in literature for PD often more extreme)
    spread1 = float(np.std(logf0) * -1.0)  # keep signed scale similar in spirit
    # spread2 ~ non-linear residual magnitude
    spread2 = float(np.mean(np.abs(logf0 - np.median(logf0))))
    return spread1, spread2


def _d2_proxy(f0: np.ndarray) -> float:
    f0 = f0[np.isfinite(f0) & (f0 > 0)]
    if f0.size < 10:
        return np.nan
    # correlation-dimension-like crude proxy: embedding variance ratio
    x = (f0 - np.mean(f0)) / (np.std(f0) + 1e-9)
    emb = np.column_stack([x[:-2], x[1:-1], x[2:]])
    # mean pairwise distance
    n = min(len(emb), 200)
    emb = emb[:n]
    dsum = 0.0
    c = 0
    for i in range(0, n, 3):
        dif = emb[i + 1 :] - emb[i]
        dsum += np.mean(np.linalg.norm(dif, axis=1))
        c += 1
    return float(dsum / max(c, 1))


def _librosa_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    import librosa

    out: Dict[str, float] = {}
    # MFCCs 1-12 (skip 0th for relative stability, but also keep 0 as energy-ish)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    for i in range(1, 13):
        out[f"MFCC_{i}"] = float(mfcc_mean[i])
        out[f"MFCC_{i}_std"] = float(mfcc_std[i])
    out["MFCC_0"] = float(mfcc_mean[0])

    # deltas
    delta = librosa.feature.delta(mfcc)
    ddelta = librosa.feature.delta(mfcc, order=2)
    out["mean_delta_log_energy"] = float(delta[0].mean())
    out["std_delta_log_energy"] = float(delta[0].std())
    out["mean_delta_delta_log_energy"] = float(ddelta[0].mean())
    out["std_delta_delta_log_energy"] = float(ddelta[0].std())

    # spectral
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    out["spectral_centroid_mean"] = float(cent.mean())
    out["spectral_bandwidth_mean"] = float(bw.mean())
    out["spectral_rolloff_mean"] = float(rolloff.mean())
    out["zcr_mean"] = float(zcr.mean())
    out["spectral_contrast_mean"] = float(contrast.mean())

    # RMS energy
    rms = librosa.feature.rms(y=y)
    out["rms_mean"] = float(rms.mean())
    out["rms_std"] = float(rms.std())
    return out


def assess_quality(y: np.ndarray, sr: int, voiced_fraction: float) -> AudioQuality:
    duration = len(y) / float(sr)
    rms = float(np.sqrt(np.mean(y**2) + 1e-12))
    snr = _snr_proxy(y)
    messages = []
    ok = True
    if duration < MIN_DURATION_SEC:
        ok = False
        messages.append(f"Recording too short ({duration:.2f}s). Please sustain voice ≥ {MIN_DURATION_SEC:.0f}s.")
    if voiced_fraction < MIN_VOICED_FRACTION:
        ok = False
        messages.append(
            f"Too little voiced speech detected ({voiced_fraction:.0%}). "
            "Please sustain a steady vowel like “aaa” closer to the microphone."
        )
    if snr < MIN_SNR_DB:
        ok = False
        messages.append(
            f"Audio seems noisy or too quiet (SNR proxy {snr:.1f} dB). "
            "Record in a quiet room and speak clearly."
        )
    if rms < 0.005:
        ok = False
        messages.append("Signal level too low. Increase microphone volume or move closer.")
    if ok:
        messages.append("Recording quality acceptable for screening analysis.")
    return AudioQuality(
        ok=ok,
        duration_sec=duration,
        rms=rms,
        snr_db_proxy=snr,
        voiced_fraction=voiced_fraction,
        messages=messages,
    )


def extract_features(
    source: Union[str, Path, bytes, bytearray],
    max_seconds: float = 12.0,
) -> ExtractionResult:
    """Full extraction pipeline from file path or raw bytes.

    max_seconds: analyze only the first N seconds after trim (stable + faster).
    """
    y, sr = _load_audio(source, sr=TARGET_SR)
    y = _trim_and_normalize(y)
    if max_seconds and len(y) > int(max_seconds * sr):
        y = y[: int(max_seconds * sr)]
    praat = _praat_voice_report(y, sr)
    libf = _librosa_features(y, sr)
    quality = assess_quality(y, sr, float(praat.get("voiced_fraction", 0.0)))

    # --- Common audio feature set (used to train Italian live model) ---
    common: Dict[str, float] = {}
    for k, v in praat.items():
        if k == "voiced_fraction":
            continue
        common[f"praat_{k}"] = float(v) if v is not None and np.isfinite(v) else np.nan
    for k, v in libf.items():
        common[k] = float(v) if v is not None and np.isfinite(v) else np.nan
    common["voiced_fraction"] = float(praat.get("voiced_fraction", 0.0))
    common["duration_sec"] = float(len(y) / sr)
    common["snr_db_proxy"] = float(quality.snr_db_proxy)

    # --- UCI-like mapping (Praat local jitter is a fraction, matching UCI scale ~0.00x) ---
    apq11 = praat.get("apq11")
    apq5 = praat.get("apq5")
    mdvp_apq = apq11 if apq11 is not None and np.isfinite(apq11) else apq5
    uci = {
        "MDVP:Fo(Hz)": praat["f0_mean"],
        "MDVP:Fhi(Hz)": praat["f0_max"],
        "MDVP:Flo(Hz)": praat["f0_min"],
        "MDVP:Jitter(%)": praat["local_jitter"],
        "MDVP:Jitter(Abs)": praat["local_abs_jitter"],
        "MDVP:RAP": praat["rap_jitter"],
        "MDVP:PPQ": praat["ppq5_jitter"],
        "Jitter:DDP": praat["ddp_jitter"],
        "MDVP:Shimmer": praat["local_shimmer"],
        "MDVP:Shimmer(dB)": praat["local_db_shimmer"],
        "Shimmer:APQ3": praat["apq3"],
        "Shimmer:APQ5": praat["apq5"],
        "MDVP:APQ": mdvp_apq,
        "Shimmer:DDA": praat["dda"],
        "NHR": praat["nhr"],
        "HNR": praat["hnr"],
        "RPDE": praat["rpde"],
        "DFA": praat["dfa"],
        "spread1": praat["spread1"],
        "spread2": praat["spread2"],
        "D2": praat["d2"],
        "PPE": praat["ppe"],
    }

    # --- BenSParX-like mapping ---
    bensparx = {
        "numPulses": praat["n_pulses"],
        "numPeriodsPulses": praat["n_periods"],
        "meanPeriodPulses": praat["mean_period"],
        "stdDevPeriodPulses": praat["std_period"],
        "locPctJitter": praat["local_jitter"],
        "locAbsJitter": praat["local_abs_jitter"],
        "rapJitter": praat["rap_jitter"],
        "ppq5Jitter": praat["ppq5_jitter"],
        "ddpJitter": praat["ddp_jitter"],
        "locShimmer": praat["local_shimmer"],
        "locDbShimmer": praat["local_db_shimmer"],
        "apq3Shimmer": praat["apq3"],
        "apq5Shimmer": praat["apq5"],
        "apq11Shimmer": praat["apq11"],
        "ddaShimmer": praat["dda"],
        "AutoCorrHarmonicity": praat["mean_autocorr_harmonicity"],
        "meanAutoCorrHarmonicity": praat["mean_autocorr_harmonicity"],
        "NHR": praat["nhr"],
        "minIntensity": praat["min_intensity"],
        "maxIntensity": praat["max_intensity"],
        "meanIntensity": praat["mean_intensity"],
        "meanPitch": praat["f0_mean"],
        "minPitch": praat["f0_min"],
        "maxPitch": praat["f0_max"],
    }
    for i in range(1, 13):
        bensparx[f"MFCC_{i}"] = libf.get(f"MFCC_{i}", np.nan)
    # extra energy deltas often present in BenSParX-style sets
    bensparx["mean_delta_log_energy"] = libf.get("mean_delta_log_energy", np.nan)
    bensparx["std_delta_log_energy"] = libf.get("std_delta_log_energy", np.nan)
    bensparx["mean_delta_delta_log_energy"] = libf.get("mean_delta_delta_log_energy", np.nan)
    bensparx["std_delta_delta_log_energy"] = libf.get("std_delta_delta_log_energy", np.nan)

    # sanitize all to float
    def _clean(d: Dict[str, float]) -> Dict[str, float]:
        return {k: (float(v) if v is not None and np.isfinite(float(v)) else np.nan) for k, v in d.items()}

    common = _clean(common)
    uci = _clean(uci)
    bensparx = _clean(bensparx)

    return ExtractionResult(
        quality=quality,
        common=common,
        uci=uci,
        bensparx=bensparx,
        common_feature_names=sorted(common.keys()),
        uci_feature_names=list(uci.keys()),
        bensparx_feature_names=list(bensparx.keys()),
        meta={"sr": sr, "n_samples": int(len(y))},
    )


def vectorize(
    feat_dict: Dict[str, float],
    feature_names: List[str],
    fill_values: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, List[str], float]:
    """
    Build ordered vector. Missing features filled from fill_values (training medians).
    Returns vector, list of imputed feature names, completeness ratio.
    """
    fill_values = fill_values or {}
    vec = []
    imputed = []
    present = 0
    for name in feature_names:
        v = feat_dict.get(name, np.nan)
        if v is None or not np.isfinite(v):
            fv = fill_values.get(name, 0.0)
            vec.append(float(fv) if fv is not None and np.isfinite(fv) else 0.0)
            imputed.append(name)
        else:
            vec.append(float(v))
            present += 1
    completeness = present / max(len(feature_names), 1)
    return np.array(vec, dtype=float), imputed, completeness
