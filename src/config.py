"""Paths and shared constants for PD voice screening."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THESIS_ROOT = ROOT.parent

DATASETS = {
    "english_uci": {
        "path": THESIS_ROOT / "datasets" / "01_UCI_Parkinsons" / "parkinsons.data",
        "language": "English",
        "display_name": "UCI Parkinsons (English)",
        "label_col": "status",
        "id_col": "name",
        "positive_label": 1,
        "positive_name": "Parkinson's Disease (PD)",
        "negative_name": "Healthy Control (HC)",
    },
    "bengali_bensparx": {
        "path": THESIS_ROOT / "datasets" / "02_BenSParX" / "BenSParX-main" / "BenSParX.csv",
        "language": "Bengali",
        "display_name": "BenSParX (Bengali)",
        "label_col": "class",
        "id_col": "id",
        "positive_label": 1,
        "positive_name": "Parkinson's Disease (PD)",
        "negative_name": "Healthy Control (HC)",
    },
}

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Clinical disclaimer shown in app and reports
DISCLAIMER = (
    "This tool is a research screening support system only. "
    "It is NOT a medical diagnosis. Final decisions must be made by qualified clinicians."
)

# Human-readable feature explanations (used for "why PD / why not")
FEATURE_GLOSSARY = {
    # UCI / common dysphonia
    "MDVP:Fo(Hz)": "Average fundamental frequency (pitch). PD speech is often softer/flatter, but average Fo alone is less reliable than variability measures.",
    "MDVP:Fhi(Hz)": "Maximum fundamental frequency in the sample. Reduced pitch range can appear in PD.",
    "MDVP:Flo(Hz)": "Minimum fundamental frequency in the sample. Limited pitch floor/range can relate to monotone speech.",
    "MDVP:Jitter(%)": "Cycle-to-cycle pitch perturbation (%). Higher jitter often indicates unstable phonation, more common in PD.",
    "MDVP:Jitter(Abs)": "Absolute jitter. Higher values suggest irregular vocal-fold vibration.",
    "MDVP:RAP": "Relative Average Perturbation of pitch. Elevated values indicate short-term pitch instability.",
    "MDVP:PPQ": "Pitch Period Perturbation Quotient. Higher values suggest irregular phonation.",
    "Jitter:DDP": "Difference of differences of periods (jitter-related). Higher = less stable pitch periods.",
    "MDVP:Shimmer": "Amplitude perturbation. Higher shimmer often relates to breathy/unstable loudness in PD.",
    "MDVP:Shimmer(dB)": "Shimmer in decibels. Higher values indicate louder-cycle instability.",
    "Shimmer:APQ3": "3-point amplitude perturbation quotient. Higher = unstable loudness.",
    "Shimmer:APQ5": "5-point amplitude perturbation quotient. Higher = unstable loudness.",
    "MDVP:APQ": "Amplitude Perturbation Quotient. Higher values suggest irregular voice intensity.",
    "Shimmer:DDA": "Average absolute differences of consecutive amplitudes. Higher = more amplitude irregularity.",
    "NHR": "Noise-to-Harmonics Ratio. Higher NHR means noisier voice quality (breathiness/hoarseness).",
    "HNR": "Harmonics-to-Noise Ratio. Lower HNR often means poorer harmonic voice quality (common in PD).",
    "RPDE": "Recurrence Period Density Entropy. Captures nonlinear irregularity of pitch periods; often higher in PD.",
    "DFA": "Detrended Fluctuation Analysis. Nonlinear signal complexity measure related to voice dynamics.",
    "spread1": "Nonlinear measure of fundamental-frequency variation (from Little et al.). Often more extreme in PD.",
    "spread2": "Nonlinear measure related to F0 variation. Helps separate PD from healthy phonation.",
    "D2": "Correlation dimension (nonlinear complexity). Differences may reflect PD-related voice dynamics.",
    "PPE": "Pitch Period Entropy. Higher PPE = less predictable/monotone-unstable pitch; strong PD cue in English data.",
    # BenSParX-style names
    "locPctJitter": "Local percent jitter. Higher values indicate short-term pitch instability.",
    "locAbsJitter": "Local absolute jitter. Higher = more irregular vocal vibration.",
    "rapJitter": "RAP jitter. Higher suggests pitch perturbation.",
    "ppq5Jitter": "5-point pitch period perturbation. Higher = less stable pitch.",
    "ddpJitter": "DDP jitter. Higher = unstable pitch periods.",
    "locShimmer": "Local shimmer. Higher = unstable loudness/amplitude.",
    "locDbShimmer": "Local shimmer in dB. Higher = amplitude irregularity.",
    "apq3Shimmer": "3-point APQ shimmer. Higher = loudness instability.",
    "apq5Shimmer": "5-point APQ shimmer. Higher = loudness instability.",
    "apq11Shimmer": "11-point APQ shimmer. Higher = longer-window amplitude irregularity.",
    "ddaShimmer": "DDA shimmer. Higher = consecutive amplitude differences.",
    "meanIntensity": "Average loudness/intensity. Lower values can reflect hypophonia (soft voice) in PD.",
    "minIntensity": "Minimum intensity. Very low minima may relate to weak voice projection.",
    "maxIntensity": "Maximum intensity. Reduced maxima can relate to limited loudness range.",
    "meanPitch": "Average pitch (normalized/dataset-specific). Useful with range features, not alone.",
    "maxPitch": "Maximum pitch. Reduced max pitch range can relate to monotone speech.",
    "minPitch": "Minimum pitch. Together with max pitch reflects pitch range.",
    "AutoCorrHarmonicity": "Harmonicity via autocorrelation. Lower harmonicity can mean noisier voice.",
    "meanAutoCorrHarmonicity": "Mean harmonicity. Lower values suggest breathier/noisier phonation.",
    "numPulses": "Number of glottal pulses detected. Extremely low/high values can flag unstable phonation.",
    "numPeriodsPulses": "Number of periods between pulses. Related to phonation continuity.",
    "meanPeriodPulses": "Mean pulse period (related to pitch period).",
    "stdDevPeriodPulses": "Std. dev. of pulse periods. Higher = less stable pitch periods.",
}


def feature_meaning(name: str) -> str:
    # Strip live-audio praat_ prefix for glossary lookup
    base = name[6:] if name.startswith("praat_") else name
    if name in FEATURE_GLOSSARY:
        return FEATURE_GLOSSARY[name]
    if base in FEATURE_GLOSSARY:
        return FEATURE_GLOSSARY[base]
    lookup = {
        "f0_mean": "Average fundamental frequency (pitch).",
        "f0_min": "Minimum pitch in the analysis window.",
        "f0_max": "Maximum pitch in the analysis window.",
        "f0_std": "Pitch variability. Reduced variation can relate to monotone PD speech.",
        "local_jitter": "Local pitch jitter. Higher often means less stable phonation.",
        "local_abs_jitter": "Absolute jitter. Higher suggests irregular vocal-fold vibration.",
        "rap_jitter": "RAP jitter (short-term pitch perturbation).",
        "ppq5_jitter": "PPQ5 pitch period perturbation.",
        "ddp_jitter": "DDP jitter measure.",
        "local_shimmer": "Local shimmer (loudness instability).",
        "local_db_shimmer": "Shimmer in dB.",
        "apq3": "3-point amplitude perturbation quotient.",
        "apq5": "5-point amplitude perturbation quotient.",
        "apq11": "11-point amplitude perturbation quotient.",
        "dda": "DDA shimmer (consecutive amplitude differences).",
        "hnr": "Harmonics-to-noise ratio. Lower often means noisier voice quality.",
        "nhr": "Noise-to-harmonics ratio. Higher often means noisier voice quality.",
        "ppe": "Pitch period entropy proxy. Higher = less predictable pitch structure.",
        "rpde": "Recurrence period density entropy proxy (pitch irregularity).",
        "dfa": "Detrended fluctuation analysis proxy of amplitude dynamics.",
        "mean_intensity": "Average loudness/intensity.",
        "min_intensity": "Minimum intensity.",
        "max_intensity": "Maximum intensity.",
        "spectral_centroid_mean": "Spectral centroid (brightness of the voice spectrum).",
        "spectral_bandwidth_mean": "Spectral bandwidth.",
        "spectral_rolloff_mean": "Spectral roll-off frequency.",
        "spectral_contrast_mean": "Spectral contrast.",
        "zcr_mean": "Zero-crossing rate (related to noisiness/periodicity).",
        "rms_mean": "Root-mean-square energy (loudness proxy).",
        "rms_std": "Energy variation over time.",
        "voiced_fraction": "Fraction of frames detected as voiced speech.",
    }
    if base in lookup:
        return f"{name}: {lookup[base]}"
    if name.startswith("MFCC_") or base.startswith("MFCC_"):
        return (
            f"{name}: Mel-frequency cepstral coefficient capturing spectral/timbre shape of speech. "
            "PD can alter articulation and spectral energy; MFCC differences support screening but are less "
            "intuitively clinical than jitter/shimmer."
        )
    if "Jitter" in name or "jitter" in name:
        return f"{name}: Pitch perturbation measure. Higher values usually mean less stable phonation."
    if "Shimmer" in name or "shimmer" in name:
        return f"{name}: Amplitude perturbation measure. Higher values usually mean less stable loudness."
    if "Intensity" in name or "intensity" in name:
        return f"{name}: Loudness-related measure. Soft voice (hypophonia) is common in PD."
    if "Pitch" in name or "pitch" in name or "Fo" in name or "F0" in name or "f0" in name:
        return f"{name}: Pitch-related measure. PD often reduces pitch variation (monotone quality)."
    if "HNR" in name or "NHR" in name or "Harmonic" in name or "hnr" in name or "nhr" in name:
        return f"{name}: Harmonic/noise quality measure. Noisier or less harmonic voice can appear in PD."
    return f"{name}: Acoustic descriptor used by the model. Compare its value against the training distribution."
