"""Checks for Primus-safe layers (no audio, isolated store, integrity hash)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.enrollment import (
    ENROLLMENT_DIR,
    enrollment_unlocks_app,
    merge_enrollment,
    parse_attestation,
)
from src.integrity import make_integrity_stamp
from src.privacy import assert_receipt_has_no_audio, make_screening_receipt
from src.primus_onchain import local_attestation_checks
from src.provenance import collect_provenance


def test_isolated_store() -> None:
    assert ".voicesense_local" in str(ENROLLMENT_DIR)
    assert "VoiceSense-PD" in str(ENROLLMENT_DIR)


def test_provenance_and_stamp() -> None:
    prov = collect_provenance()
    assert len(prov["bundle_sha256"]) == 64
    result = {
        "status": "ok",
        "language_mode": "english",
        "quality": {"ok": True},
        "primary": {
            "decision_code": "UNCERTAIN",
            "pred_label": -1,
            "probability_pd": 0.51,
            "prediction": "Uncertain",
        },
    }
    stamp = make_integrity_stamp(result, provenance_bundle_sha256=prov["bundle_sha256"])
    blob = json.dumps(stamp["public"])
    assert "0.51" not in blob
    assert "probability" not in blob
    assert stamp["primus_addition_params"]["no_audio"] is True
    receipt = make_screening_receipt(
        {**result, "warnings": [], "secondary": {}, "primary": {**result["primary"], "explanation": {}}},
        provenance={"bundle_sha256": prov["bundle_sha256"]},
        integrity=stamp,
    )
    assert_receipt_has_no_audio(receipt)
    assert receipt["integrity"]["public_hash"] == stamp["public_hash"]


def test_parse_bind_and_onchain_structure() -> None:
    att = {
        "recipient": "0x" + "11" * 20,
        "data": json.dumps({"login": "a" * 64}),
        "signatures": ["0xsig"],
        "attestors": [{"attestorAddr": "0xatt"}],
        "additionParams": json.dumps({"integrity_hash": "abc", "no_audio": True}),
    }
    rec = parse_attestation(att, purpose="uniqueness", expected_template_id="t")
    assert rec["bound_integrity_hash"] == "abc"
    merged = merge_enrollment(None, {**rec, "purpose": "researcher"})
    assert merged["researcher_ok"] is True
    checks = local_attestation_checks(att)
    assert checks["ok"] is True
    assert checks["chain_ready"] is True
    follow_att = {
        "recipient": "0x" + "11" * 20,
        "data": '{"following": true}',
        "signatures": ["0xsig"],
        "attestors": [{"attestorAddr": "0xatt"}],
    }
    frec = parse_attestation(follow_att, purpose="x_follow", expected_template_id="xf")
    assert frec["follow_ok"] is True
    nested = parse_attestation(
        {
            "recipient": "0x" + "11" * 20,
            "data": json.dumps({"data": {"user": {"result": {"legacy": {"following": True}}}}}),
            "signatures": ["0xsig"],
            "attestors": [{"attestorAddr": "0xatt"}],
        },
        purpose="x_follow",
        expected_template_id="xf",
    )
    assert nested["follow_ok"] is True
    perspectives = parse_attestation(
        {
            "recipient": "0x" + "11" * 20,
            "data": json.dumps(
                {
                    "data": {
                        "user": {
                            "result": {
                                "relationship_perspectives": {"following": True},
                                "core": {"screen_name": "its_perseus_1"},
                            }
                        }
                    }
                }
            ),
            "signatures": ["0xsig"],
            "attestors": [{"attestorAddr": "0xatt"}],
        },
        purpose="x_follow",
        expected_template_id="xf",
    )
    assert perspectives["follow_ok"] is True
    primus_flat = parse_attestation(
        {
            "recipient": "0x" + "11" * 20,
            "data": '{"screen_name":"its_perseus_1","following":"true","following.count":"1"}',
            "reponseResolve": [
                {"keyName": "following", "parsePath": "$.data.user.result.legacy.following"}
            ],
            "signatures": ["0xsig"],
            "attestors": [{"attestorAddr": "0xatt"}],
        },
        purpose="x_follow",
        expected_template_id="xf",
    )
    assert primus_flat["follow_ok"] is True
    via_extract = parse_attestation(
        {
            "recipient": "0x" + "11" * 20,
            "request": {
                "url": "https://x.com/i/api/graphql/Gb/UserByScreenName?variables=%7B%22screen_name%22%3A%22its_perseus_1%22"
            },
            "data": '{"screen_name":"its_perseus_1"}',
            "signatures": ["0xsig"],
            "attestors": [{"attestorAddr": "0xatt"}],
        },
        purpose="x_follow",
        expected_template_id="xf",
        extra_sources={"follow_extract": {"from_all_json": True, "from_data": None}},
    )
    assert via_extract["follow_ok"] is True
    screen_only = parse_attestation(
        {
            "recipient": "0x" + "11" * 20,
            "data": '{"screen_name":"its_perseus_1"}',
            "signatures": ["0xsig"],
            "attestors": [{"attestorAddr": "0xatt"}],
        },
        purpose="x_follow",
        expected_template_id="xf",
    )
    assert screen_only["follow_ok"] is not True
    trusted = parse_attestation(
        {
            "recipient": "0x" + "11" * 20,
            "data": '{"screen_name":"its_perseus_1"}',
            "signatures": ["0xsig"],
            "attestors": [{"attestorAddr": "0xatt"}],
        },
        purpose="x_follow",
        expected_template_id="xf",
        sdk_verified=True,
    )
    assert trusted["follow_ok"] is True
    count_named_following = parse_attestation(
        {
            "recipient": "0x" + "11" * 20,
            "request": {
                "url": "https://x.com/i/api/graphql/Gb/UserByScreenName?variables=%7B%22screen_name%22%3A%22its_perseus_1%22"
            },
            "data": '{"following":"652","screen_name":"its_perseus_1"}',
            "reponseResolve": [
                {"keyName": "screen_name", "parsePath": "$.data.user.result.core.screen_name"},
                {
                    "keyName": "following",
                    "parsePath": "$.data.user.result.relationship_counts.following",
                },
            ],
            "signatures": ["0xsig"],
            "attestors": [{"attestorAddr": "0xatt"}],
        },
        purpose="x_follow",
        expected_template_id="xf",
        sdk_verified=True,
        extra_sources={
            "follow_extract": {
                "from_data": None,
                "from_all_json": None,
                "from_private": None,
                "following": False,
            }
        },
    )
    assert count_named_following["follow_ok"] is True
    explicit_no = parse_attestation(
        {
            "recipient": "0x" + "11" * 20,
            "data": '{"following": false}',
            "signatures": ["0xsig"],
            "attestors": [{"attestorAddr": "0xatt"}],
        },
        purpose="x_follow",
        expected_template_id="xf",
        sdk_verified=True,
    )
    assert explicit_no["follow_ok"] is False
    merged_f = merge_enrollment(None, frec)
    assert merged_f["follow_ok"] is True
    owner_att = {
        "recipient": "0x" + "11" * 20,
        "data": json.dumps({"screen_name": "its_perseus_1", "following": True}),
        "signatures": ["0xsig"],
        "attestors": [{"attestorAddr": "0xatt"}],
    }
    orec = parse_attestation(owner_att, purpose="x_owner", expected_template_id="own")
    # Same profile-page template must not treat a follower as Perseus
    assert orec["owner_ok"] is False
    assert enrollment_unlocks_app(merge_enrollment(None, orec)) is False


if __name__ == "__main__":
    test_isolated_store()
    test_provenance_and_stamp()
    test_parse_bind_and_onchain_structure()
    print("primus layers OK")
