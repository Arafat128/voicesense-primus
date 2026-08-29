"""Optional Primus zkTLS enrollment — identity proofs only.

Never receives, stores, or hashes voice recordings. Attested Web2 fields are
reduced to fingerprints (SHA-256). Plaintext handles are stripped if a
developer forgets to set SHA256 conditions on the template.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .config import ROOT
from .privacy import utc_now_iso

ENROLLMENT_SCHEMA = "voicesense.enrollment.v1"
# Isolated from the original thesis copy (~/.voicesense)
ENROLLMENT_DIR = ROOT / ".voicesense_local"
ENROLLMENT_PATH = ENROLLMENT_DIR / "enrollment.json"
RECIPIENT_PATH = ENROLLMENT_DIR / "recipient.txt"
SID_UNLOCK_DIR = ENROLLMENT_DIR / "unlocks"
LAST_ATTESTATION_PATH = ENROLLMENT_DIR / "last_attestation_meta.json"
LAST_PARSE_DEBUG_PATH = ENROLLMENT_DIR / "last_parse_debug.json"

# Viewer-follows-profile flags. Ignore reverse / count fields.
_FOLLOW_LEAVES = frozenset(
    {"following", "is_following", "isfollowing", "viewer_following"}
)
_FOLLOW_PROXY_LEAVES = frozenset({"want_retweets"})
_FOLLOW_IGNORE_LEAVES = frozenset(
    {
        "followed_by",
        "super_following",
        "super_followed_by",
        "following_count",
        "followers_count",
        "friends_count",
        "follow_request_sent",
    }
)

HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        root = Path(__file__).resolve().parents[1]
        load_dotenv(root / ".env", override=True)
    except Exception:
        pass


def primus_settings() -> Dict[str, str]:
    _load_dotenv()
    return {
        "app_id": (os.getenv("PRIMUS_APP_ID") or "").strip(),
        "app_secret": (os.getenv("PRIMUS_APP_SECRET") or "").strip(),
        "uniqueness_template_id": (
            os.getenv("PRIMUS_UNIQUENESS_TEMPLATE_ID")
            or os.getenv("PRIMUS_TEMPLATE_ID")
            or ""
        ).strip(),
        "identity_field": (os.getenv("PRIMUS_IDENTITY_FIELD") or "screen_name").strip(),
        "age_template_id": (os.getenv("PRIMUS_AGE_TEMPLATE_ID") or "").strip(),
        "age_field": (os.getenv("PRIMUS_AGE_FIELD") or "age").strip(),
        "age_op": (os.getenv("PRIMUS_AGE_OP") or ">=").strip(),
        "age_value": (os.getenv("PRIMUS_AGE_VALUE") or "18").strip(),
        "researcher_template_id": (os.getenv("PRIMUS_RESEARCHER_TEMPLATE_ID") or "").strip(),
        "researcher_field": (os.getenv("PRIMUS_RESEARCHER_FIELD") or "email").strip(),
        "github_template_id": (os.getenv("PRIMUS_GITHUB_TEMPLATE_ID") or "").strip(),
        "github_field": (os.getenv("PRIMUS_GITHUB_FIELD") or "full_name").strip(),
        "x_follow_template_id": (os.getenv("PRIMUS_X_FOLLOW_TEMPLATE_ID") or "").strip(),
        "x_follow_field": (os.getenv("PRIMUS_X_FOLLOW_FIELD") or "following").strip(),
        "x_follow_op": (os.getenv("PRIMUS_X_FOLLOW_OP") or "=").strip() or "=",
        "x_follow_value": (os.getenv("PRIMUS_X_FOLLOW_VALUE") or "true").strip() or "true",
        "x_follow_handle": (
            os.getenv("PRIMUS_X_FOLLOW_HANDLE") or "its_perseus_1"
        ).strip().lstrip("@"),
        "owner_template_id": (
            os.getenv("PRIMUS_OWNER_TEMPLATE_ID")
            or os.getenv("PRIMUS_UNIQUENESS_TEMPLATE_ID")
            or os.getenv("PRIMUS_TEMPLATE_ID")
            or ""
        ).strip(),
        "owner_field": (os.getenv("PRIMUS_OWNER_FIELD") or "screen_name").strip(),
        "att_mode": (os.getenv("PRIMUS_ATT_MODE") or "proxytls").strip() or "proxytls",
        "require_enrollment": (os.getenv("VOICESENSE_REQUIRE_ENROLLMENT") or "1").strip()
        in ("1", "true", "TRUE", "yes"),
        "mock": (os.getenv("PRIMUS_MOCK") or "").strip() in ("1", "true", "TRUE", "yes"),
        "dev_ui": (os.getenv("VOICESENSE_DEV_UI") or "").strip() in ("1", "true", "TRUE", "yes"),
        "operator_pin": (os.getenv("VOICESENSE_OPERATOR_PIN") or "").strip(),
    }


def configured_for_live_primus() -> bool:
    s = primus_settings()
    return bool(
        s["app_id"]
        and (
            s["uniqueness_template_id"]
            or s["x_follow_template_id"]
            or s["age_template_id"]
            or s["researcher_template_id"]
            or s["github_template_id"]
            or s.get("owner_template_id")
        )
    )


def x_follow_profile_url() -> str:
    handle = primus_settings()["x_follow_handle"] or "its_perseus_1"
    return f"https://x.com/{handle}"


def enrollment_unlocks_app(record: Optional[Dict[str, Any]]) -> bool:
    """Unlock Analyze if they follow @its_perseus_1, or if they *are* that account."""
    if not record:
        return False
    return record.get("follow_ok") is True or record.get("owner_ok") is True


def local_recipient() -> str:
    """Random 0x-address used only as a Primus recipient id — not a user wallet."""
    ENROLLMENT_DIR.mkdir(parents=True, exist_ok=True)
    if RECIPIENT_PATH.exists():
        value = RECIPIENT_PATH.read_text(encoding="utf-8").strip()
        if value.startswith("0x") and len(value) == 42:
            return value
    value = "0x" + secrets.token_hex(20)
    RECIPIENT_PATH.write_text(value + "\n", encoding="utf-8")
    return value


def load_enrollment() -> Optional[Dict[str, Any]]:
    if not ENROLLMENT_PATH.exists():
        return None
    try:
        data = json.loads(ENROLLMENT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("schema") != ENROLLMENT_SCHEMA:
        return None
    return data


def save_enrollment(record: Dict[str, Any]) -> Dict[str, Any]:
    ENROLLMENT_DIR.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record["schema"] = ENROLLMENT_SCHEMA
    record.setdefault("saved_at", utc_now_iso())
    ENROLLMENT_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def clear_enrollment() -> None:
    if ENROLLMENT_PATH.exists():
        ENROLLMENT_PATH.unlink()


def save_sid_unlock(sid: str, record: Dict[str, Any]) -> None:
    """Remember a successful proof for this browser sid so a return tab can adopt it."""
    token = (sid or "").strip()
    if not token or len(token) < 16:
        return
    SID_UNLOCK_DIR.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload["schema"] = ENROLLMENT_SCHEMA
    (SID_UNLOCK_DIR / f"{token}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def load_sid_unlock(sid: str) -> Optional[Dict[str, Any]]:
    token = (sid or "").strip()
    if not token or len(token) < 16:
        return None
    path = SID_UNLOCK_DIR / f"{token}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("schema") != ENROLLMENT_SCHEMA:
        return None
    return data


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _decode_jsonish(raw: Any, *, depth: int = 4) -> Any:
    """Unwrap stringified JSON that Primus often nests one or two levels deep."""
    cur = raw
    for _ in range(depth):
        if not isinstance(cur, str):
            return cur
        text = cur.strip()
        if not text or text[0] not in "{[":
            return cur
        try:
            cur = json.loads(text)
        except json.JSONDecodeError:
            return cur
    return cur


def sanitize_attested_data(data: Any) -> Tuple[Dict[str, Any], bool]:
    """Keep hashes/booleans; replace any leftover plaintext with SHA-256."""
    data = _decode_jsonish(data)
    if not isinstance(data, dict):
        if data is None:
            return {}, False
        return {"value": _sha256_text(str(data))}, True

    sanitized: Dict[str, Any] = {}
    stripped = False
    for key, raw in data.items():
        raw = _decode_jsonish(raw)
        if isinstance(raw, bool):
            sanitized[str(key)] = raw
            continue
        if isinstance(raw, dict):
            nested, nested_stripped = sanitize_attested_data(raw)
            for nk, nv in nested.items():
                sanitized[f"{key}.{nk}"] = nv
            stripped = stripped or nested_stripped
            continue
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            sanitized[str(key)] = _sha256_text(str(raw))
            stripped = True
            continue
        text = str(raw).strip()
        lower = text.lower()
        if lower in ("true", "false"):
            sanitized[str(key)] = lower == "true"
        elif HEX64.match(text):
            sanitized[str(key)] = text.lower()
        else:
            sanitized[str(key)] = _sha256_text(text)
            stripped = True
    return sanitized, stripped


def _field_is_true(sanitized: Dict[str, Any], field: str) -> bool:
    want = (field or "following").lower()
    for key, value in sanitized.items():
        name = str(key).lower()
        if name.endswith(".count"):
            continue
        if name == want or name.endswith("." + want):
            return value is True or str(value).lower() == "true"
    return False


def _leaf_name(key: Any) -> str:
    return str(key).lower().replace("-", "_").split(".")[-1]


def _looks_like_count(value: Any) -> bool:
    """Hub often maps 'following' to relationship_counts.following (e.g. 652)."""
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value not in (0, 1)
    text = str(value).strip()
    if text.isdigit() and text not in ("0", "1"):
        return True
    return False


def _coerce_bool(value: Any) -> Optional[bool]:
    value = _decode_jsonish(value)
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if _looks_like_count(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    text = str(value).strip().lower()
    if text in ("true", "yes", "following"):
        return True
    if text in ("false", "no", "follow"):
        return False
    return None


def _iter_nodes(obj: Any, path: str = "") -> Any:
    obj = _decode_jsonish(obj)
    if isinstance(obj, dict):
        kn = obj.get("keyName") or obj.get("key_name") or obj.get("key")
        if kn is not None and ("value" in obj or "val" in obj):
            yield str(kn), obj.get("value", obj.get("val")), path
        for key, value in obj.items():
            child = f"{path}.{key}" if path else str(key)
            yield str(key), value, child
            yield from _iter_nodes(value, child)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            yield from _iter_nodes(item, f"{path}[{i}]")


def _path_is_follow_flag(key: str, path: str) -> bool:
    leaf = _leaf_name(key)
    pl = (path or "").lower()
    if "relationship_counts" in pl or "friends_count" in pl or "followers_count" in pl:
        return False
    if leaf.endswith("count") or leaf in _FOLLOW_IGNORE_LEAVES:
        return False
    if leaf in _FOLLOW_LEAVES:
        return True
    return "relationship_perspectives" in pl and pl.endswith("following")


def _raw_following_flag(data: Any) -> Optional[bool]:
    """Find viewer-follows-profile in nested GraphQL / Primus data."""
    saw_true = False
    saw_false = False
    saw_proxy = False
    for key, value, path in _iter_nodes(data):
        leaf = _leaf_name(key)
        if leaf.endswith("count"):
            continue
        if _path_is_follow_flag(key, path):
            flag = _coerce_bool(value)
            if flag is True:
                saw_true = True
            elif flag is False:
                saw_false = True
        elif leaf in _FOLLOW_PROXY_LEAVES:
            if _coerce_bool(value) is True:
                saw_proxy = True
    if saw_true:
        return True
    if saw_false:
        return False
    if saw_proxy:
        return True
    return None


def _flatten_resolves(raw: Any) -> list:
    if raw is None:
        return []
    raw = _decode_jsonish(raw)
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: list = []
    for item in raw:
        item = _decode_jsonish(item)
        if not isinstance(item, dict):
            continue
        nested = item.get("oneUrlResponseResolve") or item.get("one_url_response_resolve")
        if nested:
            out.extend(_flatten_resolves(nested))
        else:
            out.append(item)
    return out


def _follow_from_resolves(att: Dict[str, Any], decoded_data: Any) -> Optional[bool]:
    """Map Hub parsePath/keyName onto attestation.data."""
    resolves = _flatten_resolves(
        att.get("reponseResolve") or att.get("responseResolve") or att.get("responseResolves")
    )
    decoded = decoded_data if isinstance(decoded_data, dict) else {}
    result: Optional[bool] = None
    for item in resolves:
        path = str(item.get("parsePath") or item.get("parse_path") or "").lower()
        name = str(item.get("keyName") or item.get("key_name") or "")
        if "followed_by" in path or "following_count" in path or "relationship_counts" in path:
            continue
        leaf = path.split(".")[-1] if path else _leaf_name(name)
        looks_follow = leaf == "following" or path.endswith(".following") or (
            "relationship_perspectives" in path and "following" in path
        )
        if not looks_follow and _leaf_name(name) not in _FOLLOW_LEAVES:
            continue
        value = decoded.get(name) if name else None
        if value is None and isinstance(decoded, dict):
            for dk, dv in decoded.items():
                if str(dk).lower() == name.lower() or str(dk).lower().endswith("." + name.lower()):
                    value = dv
                    break
        flag = _coerce_bool(value)
        if flag is True:
            return True
        if flag is False:
            result = False
    return result


def _request_mentions_handle(att: Dict[str, Any], handle: str) -> bool:
    handle = (handle or "").strip().lstrip("@").lower()
    if not handle:
        return False
    req = att.get("request")
    urls: list = []
    if isinstance(req, dict):
        urls.append(str(req.get("url") or ""))
    elif isinstance(req, list):
        for item in req:
            if isinstance(item, dict):
                urls.append(str(item.get("url") or ""))
            else:
                urls.append(str(item))
    blob = " ".join(urls).lower()
    return handle in blob or "userbyscreenname" in blob.replace("_", "")


def attested_field_names(data: Any, att: Optional[Dict[str, Any]] = None) -> list:
    names = []
    seen = set()
    decoded = _decode_jsonish(data)
    if isinstance(decoded, dict):
        for key in decoded.keys():
            k = str(key)
            if k not in seen:
                seen.add(k)
                names.append(k)
    if att:
        for item in _flatten_resolves(
            att.get("reponseResolve") or att.get("responseResolve")
        ):
            kn = item.get("keyName") or item.get("key_name")
            if kn and str(kn) not in seen:
                seen.add(str(kn))
                names.append(str(kn))
    return names


def save_parse_debug(meta: Dict[str, Any]) -> None:
    """Keys / booleans only — never store raw handle or audio."""
    ENROLLMENT_DIR.mkdir(parents=True, exist_ok=True)
    safe = {
        "saved_at": utc_now_iso(),
        "purpose": meta.get("purpose"),
        "follow_ok": meta.get("follow_ok"),
        "owner_ok": meta.get("owner_ok"),
        "field_names": meta.get("field_names") or [],
        "resolve_paths": meta.get("resolve_paths") or [],
        "follow_extract": meta.get("follow_extract"),
        "request_mentions_handle": meta.get("request_mentions_handle"),
        "error": meta.get("error"),
    }
    LAST_PARSE_DEBUG_PATH.write_text(json.dumps(safe, indent=2) + "\n", encoding="utf-8")


def try_operator_unlock(pin: str) -> bool:
    expected = (os.getenv("VOICESENSE_OPERATOR_PIN") or "").strip()
    offered = (pin or "").strip()
    if not expected or not offered:
        return False
    return secrets.compare_digest(offered, expected)


def operator_unlock_record() -> Dict[str, Any]:
    return {
        "schema": ENROLLMENT_SCHEMA,
        "purpose": "x_owner",
        "identity_fingerprint": _sha256_text("operator|" + utc_now_iso()),
        "age_ok": None,
        "follow_ok": None,
        "owner_ok": True,
        "template_id": "operator-pin",
        "sanitized_fields": [],
        "plaintext_stripped": False,
        "sdk_verified": False,
        "source": "operator-pin",
        "mock": False,
        "recipient": local_recipient(),
        "timestamp": "",
        "verified_at": utc_now_iso(),
        "no_raw_identity": True,
        "no_audio": True,
        "enrolled": True,
    }


def fingerprint_from_sanitized(sanitized: Dict[str, Any], template_id: str = "") -> str:
    payload = json.dumps(
        {"template_id": template_id, "data": sanitized},
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_text(payload)


def _extract_attestation_object(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError("Attestation must be a JSON object")
    if "attestation" in payload and isinstance(payload["attestation"], dict):
        inner = dict(payload["attestation"])
        for k in ("attestor", "signature", "signatures", "taskId"):
            if k in payload and k not in inner:
                inner[k] = payload[k]
        return inner
    return payload


def parse_attestation(
    payload: Any,
    *,
    expected_template_id: Optional[str] = None,
    source: str = "paste",
    sdk_verified: bool = False,
    purpose: str = "uniqueness",
    extra_sources: Optional[Any] = None,
) -> Dict[str, Any]:
    att = _extract_attestation_object(payload)
    data = att.get("data")
    sanitized, stripped = sanitize_attested_data(data)

    template_id = (
        expected_template_id
        or att.get("templateId")
        or att.get("template_id")
        or ""
    )
    fingerprint = fingerprint_from_sanitized(sanitized, template_id)

    signatures = att.get("signatures") or []
    if att.get("signature") and not signatures:
        signatures = [att.get("signature")]

    if not sdk_verified and source == "paste":
        if not signatures and not att.get("attestors"):
            raise ValueError(
                "This JSON does not look like a Primus attestation "
                "(missing signatures/attestors)."
            )

    age_ok: Optional[bool] = None
    follow_ok: Optional[bool] = None
    owner_ok: Optional[bool] = None
    if purpose == "age":
        bools = [v for v in sanitized.values() if isinstance(v, bool)]
        if bools:
            age_ok = all(bools)
        elif any(str(v).lower() == "true" for v in sanitized.values()):
            age_ok = True
        elif sanitized:
            age_ok = False
    elif purpose == "x_follow":
        settings = primus_settings()
        follow_ok = True if _field_is_true(
            sanitized, settings.get("x_follow_field") or "following"
        ) else None
        if follow_ok is not True:
            nested = _raw_following_flag(data)
            if nested is True:
                follow_ok = True
            elif nested is False:
                follow_ok = False
        if follow_ok is not True:
            resolved = _follow_from_resolves(att, _decode_jsonish(data))
            if resolved is True:
                follow_ok = True
            elif resolved is False and follow_ok is not True:
                follow_ok = False
        extras = extra_sources if isinstance(extra_sources, dict) else {}
        extract = extras.get("follow_extract") if isinstance(extras.get("follow_extract"), dict) else {}
        extract_walk = dict(extract)
        # Client sets following:false when it did not find a boolean. That is
        # "unknown", not "user does not follow" — and Hub often names a COUNT
        # field `following`.
        if (
            extract.get("from_data") is None
            and extract.get("from_all_json") is None
            and extract.get("from_private") is None
        ):
            extract_walk.pop("following", None)
        if follow_ok is not True:
            for blob in (
                extras.get("all_json_response"),
                extras.get("private_data"),
                extras.get("extended_data"),
                extract_walk,
            ):
                if blob is None:
                    continue
                nested = _raw_following_flag(blob)
                if nested is True:
                    follow_ok = True
                    break
                if nested is False and follow_ok is None:
                    follow_ok = False
        if follow_ok is not True:
            for hint_key in ("from_data", "from_all_json", "from_private"):
                if _coerce_bool(extract.get(hint_key)) is True:
                    follow_ok = True
                    break
        # Hub UserByScreenName templates often attest screen_name only, or the
        # profile's following-count. Extension Success of this template unlocks
        # unless the payload explicitly says following=false.
        if follow_ok is not True and follow_ok is not False and sdk_verified:
            follow_ok = True
    elif purpose == "x_owner":
        settings = primus_settings()
        # The follow template is UserByScreenName of @its_perseus_1. Its screen_name
        # is always that handle, so it cannot prove the viewer is Perseus.
        same_tpl = bool(
            settings.get("owner_template_id")
            and settings.get("x_follow_template_id")
            and settings["owner_template_id"] == settings["x_follow_template_id"]
        )
        owner_ok = False
        if not same_tpl:
            bools = [v for v in sanitized.values() if isinstance(v, bool)]
            owner_ok = (len(bools) == 1 and bools[0] is True) or _field_is_true(
                sanitized, settings.get("owner_field") or "screen_name"
            )

    record: Dict[str, Any] = {
        "schema": ENROLLMENT_SCHEMA,
        "purpose": purpose,
        "identity_fingerprint": fingerprint,
        "age_ok": age_ok,
        "follow_ok": follow_ok,
        "owner_ok": owner_ok,
        "template_id": template_id,
        "sanitized_fields": sorted(sanitized.keys()),
        "attested_field_names": attested_field_names(data, att),
        "plaintext_stripped": stripped,
        "sdk_verified": bool(sdk_verified),
        "source": source,
        "mock": False,
        "recipient": att.get("recipient") or "",
        "timestamp": att.get("timestamp") or "",
        "verified_at": utc_now_iso(),
        "no_raw_identity": True,
        "no_audio": True,
        "bound_integrity_hash": None,
    }
    extra_raw = att.get("additionParams") or att.get("addition_params") or ""
    if extra_raw:
        try:
            extra = json.loads(extra_raw) if isinstance(extra_raw, str) else extra_raw
            if isinstance(extra, dict):
                record["bound_integrity_hash"] = extra.get("integrity_hash")
                record["bound_model_bundle"] = extra.get("model_bundle_sha256")
        except (TypeError, json.JSONDecodeError):
            pass
    return record


def merge_enrollment(
    existing: Optional[Dict[str, Any]],
    incoming: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine uniqueness + age proofs without keeping raw identity."""
    base = dict(existing or {})
    purpose = incoming.get("purpose") or "uniqueness"
    if incoming.get("bound_integrity_hash"):
        base["bound_integrity_hash"] = incoming.get("bound_integrity_hash")
        base["bound_model_bundle"] = incoming.get("bound_model_bundle")
    if purpose == "age":
        base["age_ok"] = incoming.get("age_ok")
        base["age_template_id"] = incoming.get("template_id")
        base["age_verified_at"] = incoming.get("verified_at")
        base["age_source"] = incoming.get("source")
        base["age_mock"] = bool(incoming.get("mock"))
    elif purpose == "x_follow":
        base["follow_ok"] = incoming.get("follow_ok")
        base["follow_template_id"] = incoming.get("template_id")
        base["follow_verified_at"] = incoming.get("verified_at")
        base["follow_source"] = incoming.get("source")
        if incoming.get("identity_fingerprint"):
            base["identity_fingerprint"] = incoming.get("identity_fingerprint")
    elif purpose == "x_owner":
        base["owner_ok"] = incoming.get("owner_ok")
        base["owner_template_id"] = incoming.get("template_id")
        base["owner_verified_at"] = incoming.get("verified_at")
        if incoming.get("identity_fingerprint"):
            base["identity_fingerprint"] = incoming.get("identity_fingerprint")
    elif purpose == "researcher":
        base["researcher_ok"] = True
        base["researcher_fingerprint"] = incoming.get("identity_fingerprint")
        base["researcher_template_id"] = incoming.get("template_id")
        base["researcher_verified_at"] = incoming.get("verified_at")
        base["researcher_source"] = incoming.get("source")
    elif purpose == "github_repo":
        base["github_repo_ok"] = True
        base["github_repo_fingerprint"] = incoming.get("identity_fingerprint")
        base["github_template_id"] = incoming.get("template_id")
        base["github_verified_at"] = incoming.get("verified_at")
    else:
        for key in (
            "identity_fingerprint",
            "template_id",
            "sanitized_fields",
            "plaintext_stripped",
            "sdk_verified",
            "source",
            "recipient",
            "timestamp",
            "mock",
        ):
            if key in incoming:
                base[key] = incoming[key]
        base["verified_at"] = incoming.get("verified_at")
        if incoming.get("age_ok") is not None and base.get("age_ok") is None:
            base["age_ok"] = incoming.get("age_ok")

    base["schema"] = ENROLLMENT_SCHEMA
    base["no_raw_identity"] = True
    base["no_audio"] = True
    base["enrolled"] = bool(
        base.get("identity_fingerprint") or base.get("follow_ok") or base.get("owner_ok")
    )
    return base


def mock_owner_record() -> Dict[str, Any]:
    token = secrets.token_hex(16)
    return {
        "schema": ENROLLMENT_SCHEMA,
        "purpose": "x_owner",
        "identity_fingerprint": _sha256_text("mock-owner|" + token),
        "age_ok": None,
        "follow_ok": None,
        "owner_ok": True,
        "template_id": "mock-x-owner",
        "sanitized_fields": ["screen_name"],
        "plaintext_stripped": False,
        "sdk_verified": False,
        "source": "mock",
        "mock": True,
        "recipient": local_recipient(),
        "timestamp": "",
        "verified_at": utc_now_iso(),
        "no_raw_identity": True,
        "no_audio": True,
        "enrolled": True,
    }


def mock_follow_record() -> Dict[str, Any]:
    token = secrets.token_hex(32)
    return {
        "schema": ENROLLMENT_SCHEMA,
        "purpose": "x_follow",
        "identity_fingerprint": _sha256_text("mock-follow|" + token),
        "age_ok": None,
        "follow_ok": True,
        "template_id": "mock-x-follow",
        "sanitized_fields": ["following"],
        "plaintext_stripped": False,
        "sdk_verified": False,
        "source": "mock",
        "mock": True,
        "recipient": local_recipient(),
        "timestamp": "",
        "verified_at": utc_now_iso(),
        "no_raw_identity": True,
        "no_audio": True,
        "enrolled": True,
    }


def mock_uniqueness_record() -> Dict[str, Any]:
    token = secrets.token_hex(32)
    return {
        "schema": ENROLLMENT_SCHEMA,
        "purpose": "uniqueness",
        "identity_fingerprint": _sha256_text("mock|" + token),
        "age_ok": None,
        "template_id": "mock",
        "sanitized_fields": ["mock"],
        "plaintext_stripped": False,
        "sdk_verified": False,
        "source": "mock",
        "mock": True,
        "recipient": local_recipient(),
        "timestamp": "",
        "verified_at": utc_now_iso(),
        "no_raw_identity": True,
        "no_audio": True,
        "enrolled": True,
    }


def save_attestation_meta(meta: Dict[str, Any]) -> None:
    """Store signature/attestor metadata only — never WAV, never plaintext identity."""
    ENROLLMENT_DIR.mkdir(parents=True, exist_ok=True)
    safe = {
        "saved_at": utc_now_iso(),
        "purpose": meta.get("purpose"),
        "sdk_verified": bool(meta.get("sdk_verified")),
        "has_signatures": bool(meta.get("has_signatures")),
        "attestor_count": meta.get("attestor_count"),
        "recipient": meta.get("recipient") or "",
        "chain_ready": bool(meta.get("chain_ready")),
        "issues": meta.get("issues") or [],
        "bound_integrity_hash": meta.get("bound_integrity_hash"),
    }
    LAST_ATTESTATION_PATH.write_text(json.dumps(safe, indent=2) + "\n", encoding="utf-8")


def public_enrollment_view(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not record:
        return None
    return {
        "enrolled": bool(
            record.get("identity_fingerprint")
            or record.get("follow_ok")
            or record.get("owner_ok")
        ),
        "identity_fingerprint": record.get("identity_fingerprint"),
        "age_ok": record.get("age_ok"),
        "follow_ok": record.get("follow_ok"),
        "owner_ok": record.get("owner_ok"),
        "source": record.get("source"),
        "mock": bool(record.get("mock")),
        "verified_at": record.get("verified_at"),
        "plaintext_stripped": bool(record.get("plaintext_stripped")),
        "sdk_verified": bool(record.get("sdk_verified")),
        "researcher_ok": bool(record.get("researcher_ok")),
        "github_repo_ok": bool(record.get("github_repo_ok")),
        "bound_integrity_hash": record.get("bound_integrity_hash"),
    }
