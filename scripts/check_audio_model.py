import joblib
import pandas as pd

p = joblib.load("models/live_audio_italian__random_forest.joblib")
print("top features:")
for x in (p.get("feature_importances") or [])[:20]:
    print(f"  {x['importance']:.4f}  {x['feature']}")

df = pd.read_csv("models/italian_common_features_cache.csv")
print("subjects", df.subject.nunique(), "labels", df.label.value_counts().to_dict())
print("duration by label", df.groupby("label")["duration_sec"].mean().to_dict())
print("snr by label", df.groupby("label")["snr_db_proxy"].mean().to_dict())
print("hnr by label", df.groupby("label")["praat_hnr"].mean().to_dict())
print("jitter by label", df.groupby("label")["praat_local_jitter"].mean().to_dict())
print("shimmer by label", df.groupby("label")["praat_local_shimmer"].mean().to_dict())
