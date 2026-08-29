from pathlib import Path
from src.predict_audio import predict_from_audio

root = Path(r"D:\fugu\Thesis\datasets\07_Italian_PD_Voice\italian_parkinson")
pd_wav = next((root / "28 People with Parkinson's disease").rglob("*.wav"))
hc_wav = next((root / "22 Elderly Healthy Control").rglob("*.wav"))

for label, path in [("PD", pd_wav), ("HC", hc_wav)]:
    print("=" * 60)
    print(label, path.name)
    r = predict_from_audio(path, language_mode="english", require_quality_ok=True)
    print("status", r["status"])
    if r["status"] != "ok":
        print(r)
        continue
    p = r["primary"]
    print("pred", p["prediction"], "p_pd", round(p["probability_pd"], 3))
    print("quality", r["quality"]["ok"], "voiced", round(r["quality"]["voiced_fraction"], 3))
    print("why_pd", len(p["explanation"]["why_pd"]), "why_hc", len(p["explanation"]["why_not_pd"]))
    if r.get("secondary"):
        for k, v in r["secondary"].items():
            print(" sec", k, v["prediction"], round(v["probability_pd"], 3), "comp", round(v["completeness"], 2))
print("DONE")
