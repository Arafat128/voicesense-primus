"""Sanity checks for receipts and enrollment fingerprinting (no audio models)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.enrollment import (
    fingerprint_from_sanitized,
    merge_enrollment,
    mock_uniqueness_record,
    parse_attestation,
    sanitize_attested_data,
)
from src.privacy import assert_receipt_has_no_audio, make_screening_receipt, receipt_json


def test_receipt_omits_audio() -> None:
    result = {
        "status": "ok",
        "language_mode": "english",
        "quality": {"ok": True, "duration_sec": 6.2},
        "primary": {
            "prediction": "Uncertain — not enough evidence to flag PD",
            "decision_code": "UNCERTAIN",
            "pred_label": -1,
            "probability_pd": 0.51,
            "probability_hc": 0.49,
            "model_name": "random_forest",
            "display_name": "live",
            "explanation": {
                "decision": "Uncertain",
                "summary_lines": ["ok"],
                "why_pd": [{"feature": "praat_local_jitter", "comparison": "higher"}],
                "top_influential_features": [{"feature": "praat_local_jitter", "value": 0.01}],
            },
            "imputed_features": ["secret"],
        },
        "secondary": {},
        "warnings": [],
        "extraction_preview": {"common_top": {"praat_local_jitter": 0.01}},
    }
    receipt = make_screening_receipt(
        result,
        source_kind="upload:patient-name.wav",
        include_features=False,
        enrollment={"identity_fingerprint": "abc", "mock": False, "source": "primus-zktls"},
        local_runtime=True,
    )
    assert_receipt_has_no_audio(receipt)
    blob = receipt_json(receipt)
    assert "patient-name.wav" not in blob
    assert receipt["source_kind"] == "upload"
    assert receipt["audio_retained"] is False
    assert receipt["privacy"]["raw_audio_included"] is False
    assert "imputed_features" not in json.dumps(receipt["primary"])
    assert "top_influential_features" not in blob
    assert receipt["enrollment"]["identity_fingerprint"] == "abc"


def test_sanitize_strips_handle() -> None:
    sanitized, stripped = sanitize_attested_data({"screen_name": "alice_research"})
    assert stripped is True
    assert sanitized["screen_name"] != "alice_research"
    assert len(sanitized["screen_name"]) == 64
    hashed, stripped2 = sanitize_attested_data(
        {"screen_name": "a" * 64}
    )
    # 64 hex a's look like a hash already
    assert stripped2 is False
    assert hashed["screen_name"] == "a" * 64


def test_parse_attestation_and_mock() -> None:
    att = {
        "recipient": "0x" + "11" * 20,
        "data": json.dumps({"screen_name": "bob"}),
        "timestamp": 1,
        "signatures": ["0xsig"],
        "attestors": [{"attestorAddr": "0xatt", "url": "https://primuslabs.xyz"}],
    }
    rec = parse_attestation(att, expected_template_id="tmpl", purpose="uniqueness")
    assert rec["no_audio"] is True
    assert rec["plaintext_stripped"] is True
    assert "bob" not in json.dumps(rec)
    assert rec["identity_fingerprint"] == fingerprint_from_sanitized(
        sanitize_attested_data(json.loads(att["data"]))[0],
        "tmpl",
    )
    mock = mock_uniqueness_record()
    merged = merge_enrollment(rec, {**mock, "purpose": "uniqueness"})
    assert merged["mock"] is True
    age = parse_attestation(
        {
            "data": '{"eligible": true}',
            "signatures": ["0x"],
            "attestors": [{}],
        },
        purpose="age",
    )
    merged = merge_enrollment(merged, age)
    assert merged["age_ok"] is True
    assert merged["identity_fingerprint"]


if __name__ == "__main__":
    test_receipt_omits_audio()
    test_sanitize_strips_handle()
    test_parse_attestation_and_mock()
    print("privacy checks OK")
