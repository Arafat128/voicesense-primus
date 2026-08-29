"""Optional Primus attestation checks (local + contract addresses).

On-chain verifyAttestation needs a wallet/RPC and the full ABI. This module
validates structure locally and documents official verifier addresses so a
later contract call can be added without guessing.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# From Primus docs: enterprise on-chain interaction overview
PRIMUS_VERIFIER = {
    8453: "0xCE7cefB3B5A7eB44B59F60327A53c9Ce53B0afdE",  # Base
    84532: "0xCE7cefB3B5A7eB44B59F60327A53c9Ce53B0afdE",  # Base Sepolia
    56: "0xF3C20A5216d669C521ffe3724C1439aE0897aC33",  # BNB
    42161: "0x982Cef8d9F184566C2BeC48c4fb9b6e7B0b4A58B",  # Arbitrum
}


def local_attestation_checks(attestation: Any) -> Dict[str, Any]:
    issues: List[str] = []
    obj = attestation
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if not isinstance(obj, dict):
        return {"ok": False, "issues": ["not a JSON object"], "chain_ready": False}

    inner = obj.get("attestation") if isinstance(obj.get("attestation"), dict) else obj
    signatures = inner.get("signatures") or obj.get("signatures") or []
    if obj.get("signature") and not signatures:
        signatures = [obj.get("signature")]
    attestors = inner.get("attestors") or obj.get("attestors") or []
    data = inner.get("data")
    recipient = inner.get("recipient") or obj.get("recipient")

    if not signatures:
        issues.append("missing signatures")
    if not attestors and not obj.get("attestor"):
        issues.append("missing attestors")
    if data in (None, "", {}):
        issues.append("missing data")
    if not recipient:
        issues.append("missing recipient")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "has_signatures": bool(signatures),
        "attestor_count": len(attestors) if isinstance(attestors, list) else 0,
        "recipient": recipient or "",
        "chain_ready": len(issues) == 0,
        "suggested_verifiers": PRIMUS_VERIFIER,
        "note": (
            "Local structure check only. Submit the same attestation to "
            "IPrimusZKTLS.verifyAttestation on a Primus verifier contract "
            "if you need on-chain settlement. Never put P(PD) in that payload."
        ),
    }


def verifier_for_chain(chain_id: int) -> Optional[str]:
    return PRIMUS_VERIFIER.get(int(chain_id))
