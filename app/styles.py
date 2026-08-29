"""Premium production UI styles for VoiceSense PD Screening."""

PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Instrument+Serif:ital@0;1&display=swap');

:root {
  --bg0: #070b14;
  --bg1: #0d1526;
  --card: rgba(16, 24, 40, 0.72);
  --card-border: rgba(148, 163, 184, 0.14);
  --text: #e8eef9;
  --muted: #94a3b8;
  --accent: #38bdf8;
  --accent2: #818cf8;
  --success: #34d399;
  --warn: #fbbf24;
  --danger: #fb7185;
  --glow: rgba(56, 189, 248, 0.25);
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg0) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', system-ui, sans-serif !important;
}

/* Animated premium background */
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(56,189,248,0.18), transparent 55%),
    radial-gradient(900px 500px at 90% 10%, rgba(129,140,248,0.16), transparent 50%),
    radial-gradient(800px 500px at 50% 100%, rgba(52,211,153,0.08), transparent 45%),
    linear-gradient(165deg, #070b14 0%, #0b1220 45%, #0a1020 100%) !important;
  background-attachment: fixed !important;
}

[data-testid="stAppViewContainer"]::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  opacity: 0.35;
  background-image:
    linear-gradient(rgba(148,163,184,0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(148,163,184,0.05) 1px, transparent 1px);
  background-size: 48px 48px;
  mask-image: radial-gradient(ellipse at center, black 20%, transparent 75%);
}

[data-testid="stHeader"] {
  background: rgba(7, 11, 20, 0.55) !important;
  backdrop-filter: blur(12px);
  border-bottom: 1px solid rgba(148,163,184,0.08);
}

[data-testid="stToolbar"] { background: transparent !important; }

.block-container {
  padding-top: 1.4rem !important;
  padding-bottom: 3rem !important;
  max-width: 1180px !important;
  position: relative;
  z-index: 1;
}

/* Hide streamlit chrome noise */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header [data-testid="stDecoration"] { display: none; }

/* Hero */
.vs-hero {
  position: relative;
  border-radius: 24px;
  padding: 2rem 2.1rem 1.7rem;
  margin-bottom: 1.25rem;
  overflow: hidden;
  border: 1px solid var(--card-border);
  background:
    linear-gradient(135deg, rgba(16,24,40,0.88), rgba(12,18,32,0.75));
  box-shadow:
    0 0 0 1px rgba(56,189,248,0.06),
    0 24px 80px rgba(0,0,0,0.45),
    inset 0 1px 0 rgba(255,255,255,0.04);
}
.vs-hero::after {
  content: "";
  position: absolute;
  width: 280px; height: 280px;
  right: -60px; top: -80px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(56,189,248,0.28), transparent 70%);
  filter: blur(8px);
  pointer-events: none;
}
.vs-kicker {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  font-size: 0.78rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
  font-weight: 600;
  margin-bottom: 0.65rem;
}
.vs-kicker span {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--success);
  box-shadow: 0 0 12px var(--success);
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.55; transform: scale(0.85); }
}
.vs-title {
  font-family: 'Instrument Serif', Georgia, serif;
  font-size: clamp(2rem, 4vw, 2.85rem);
  line-height: 1.1;
  margin: 0 0 0.65rem 0;
  color: #f8fafc;
  font-weight: 400;
}
.vs-title em {
  font-style: italic;
  background: linear-gradient(90deg, #7dd3fc, #a5b4fc, #6ee7b7);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.vs-sub {
  color: var(--muted);
  font-size: 1.02rem;
  max-width: 62ch;
  line-height: 1.55;
  margin: 0;
}
.vs-badges {
  display: flex; flex-wrap: wrap; gap: 0.5rem;
  margin-top: 1.15rem;
}
.vs-badge {
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.03em;
  padding: 0.35rem 0.7rem;
  border-radius: 999px;
  border: 1px solid rgba(148,163,184,0.18);
  background: rgba(15,23,42,0.55);
  color: #cbd5e1;
}

/* Glass cards */
.vs-card {
  border-radius: 18px;
  padding: 1.15rem 1.25rem;
  border: 1px solid var(--card-border);
  background: var(--card);
  backdrop-filter: blur(16px);
  box-shadow: 0 12px 40px rgba(0,0,0,0.28);
  margin-bottom: 0.85rem;
}
.vs-card h3, .vs-card h4 {
  margin-top: 0;
  color: #f1f5f9;
}
.vs-muted { color: var(--muted); font-size: 0.92rem; line-height: 1.5; }

/* Result pills */
.vs-result {
  border-radius: 18px;
  padding: 1.25rem 1.4rem;
  border: 1px solid var(--card-border);
  margin: 0.75rem 0 1rem;
  position: relative;
  overflow: hidden;
}
.vs-result.pd {
  background: linear-gradient(135deg, rgba(251,113,133,0.16), rgba(15,23,42,0.7));
  border-color: rgba(251,113,133,0.35);
  box-shadow: 0 0 40px rgba(251,113,133,0.12);
}
.vs-result.hc {
  background: linear-gradient(135deg, rgba(52,211,153,0.14), rgba(15,23,42,0.7));
  border-color: rgba(52,211,153,0.32);
  box-shadow: 0 0 40px rgba(52,211,153,0.1);
}
.vs-result.unc {
  background: linear-gradient(135deg, rgba(251,191,36,0.14), rgba(15,23,42,0.7));
  border-color: rgba(251,191,36,0.32);
  box-shadow: 0 0 40px rgba(251,191,36,0.1);
}
.vs-result .label {
  font-size: 0.75rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 600;
}
.vs-result .value {
  font-family: 'Instrument Serif', Georgia, serif;
  font-size: 1.75rem;
  margin-top: 0.25rem;
  color: #fff;
}

/* Metrics polish */
[data-testid="stMetric"] {
  background: rgba(15,23,42,0.55);
  border: 1px solid rgba(148,163,184,0.12);
  border-radius: 14px;
  padding: 0.75rem 0.9rem;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; }
[data-testid="stMetricValue"] { color: #f8fafc !important; }

/* Buttons */
.stButton > button {
  border-radius: 12px !important;
  border: 1px solid rgba(56,189,248,0.35) !important;
  background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
  color: white !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em;
  box-shadow: 0 10px 30px rgba(14,165,233,0.25) !important;
  transition: transform .15s ease, box-shadow .15s ease !important;
}
.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 14px 36px rgba(99,102,241,0.35) !important;
}
.stButton > button[kind="secondary"] {
  background: rgba(15,23,42,0.8) !important;
  border: 1px solid rgba(148,163,184,0.25) !important;
  box-shadow: none !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.4rem;
  background: rgba(15,23,42,0.45);
  padding: 0.35rem;
  border-radius: 14px;
  border: 1px solid rgba(148,163,184,0.12);
}
.stTabs [data-baseweb="tab"] {
  border-radius: 10px;
  color: var(--muted);
  font-weight: 600;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(14,165,233,0.25), rgba(99,102,241,0.22)) !important;
  color: #f8fafc !important;
}

/* Inputs */
[data-testid="stFileUploader"],
.stRadio, .stSelectbox, .stCheckbox {
  color: var(--text);
}
div[data-baseweb="select"] > div,
.stTextInput input, .stNumberInput input {
  background: rgba(15,23,42,0.75) !important;
  border-color: rgba(148,163,184,0.2) !important;
  color: var(--text) !important;
  border-radius: 12px !important;
}

/* Expanders / alerts */
[data-testid="stExpander"] {
  background: rgba(15,23,42,0.45);
  border: 1px solid rgba(148,163,184,0.12);
  border-radius: 12px;
}
.stAlert {
  border-radius: 14px !important;
  border: 1px solid rgba(148,163,184,0.15) !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid rgba(148,163,184,0.12);
}

.vs-footer {
  margin-top: 2rem;
  padding: 1rem 0 0.25rem;
  border-top: 1px solid rgba(148,163,184,0.1);
  color: var(--muted);
  font-size: 0.82rem;
  text-align: center;
}

/* Soft section label */
.vs-section {
  font-size: 0.8rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent);
  font-weight: 700;
  margin: 0.4rem 0 0.75rem;
}

.vs-privacy {
  border-radius: 16px;
  padding: 0.9rem 1.1rem;
  margin: 0.35rem 0 1rem;
  font-size: 0.92rem;
  line-height: 1.5;
  color: #e2e8f0;
}
.vs-privacy strong { color: #f8fafc; }
.vs-privacy code {
  background: rgba(15,23,42,0.75);
  padding: 0.08rem 0.32rem;
  border-radius: 6px;
  font-size: 0.86em;
}
.vs-privacy.ok {
  background: linear-gradient(135deg, rgba(52,211,153,0.14), rgba(15,23,42,0.55));
  border: 1px solid rgba(52,211,153,0.28);
}
.vs-privacy.warn {
  background: linear-gradient(135deg, rgba(251,191,36,0.14), rgba(15,23,42,0.55));
  border: 1px solid rgba(251,191,36,0.3);
}
</style>
"""
