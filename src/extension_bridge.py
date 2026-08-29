"""Local page so the Primus Chrome extension can run outside Streamlit's iframe."""
from __future__ import annotations

import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from .enrollment import ENROLLMENT_DIR, local_recipient

FRONTEND = Path(__file__).resolve().parents[1] / "app" / "primus_enroll" / "frontend"
CONFIG_PATH = ENROLLMENT_DIR / "prove_config.json"
RESULT_PATH = ENROLLMENT_DIR / "extension_result.json"
LAST_RESULT_PATH = ENROLLMENT_DIR / "last_extension_result.json"
LAST_FOLLOW_LOG_PATH = ENROLLMENT_DIR / "last_follow_log.json"
PORT = 8765

_lock = threading.Lock()
_started = False


class _Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND), **kwargs)

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        return

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, payload: Any, status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self._cors()
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/prove-config.json", "/prove-config"):
            if CONFIG_PATH.exists():
                self._send_json(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
            else:
                self._send_json({"error": "no config"}, 404)
            return
        if path in ("/extension_result.json", "/result"):
            if RESULT_PATH.exists():
                self._send_json(json.loads(RESULT_PATH.read_text(encoding="utf-8")))
            else:
                self._send_json({"ok": False, "empty": True})
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != "/submit":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "invalid json"}, 400)
            return
        ENROLLMENT_DIR.mkdir(parents=True, exist_ok=True)
        blob = json.dumps(payload, indent=2) + "\n"
        RESULT_PATH.write_text(blob, encoding="utf-8")
        LAST_RESULT_PATH.write_text(blob, encoding="utf-8")
        att = payload.get("attestation") if isinstance(payload, dict) else {}
        data = att.get("data") if isinstance(att, dict) else None
        keys = payload.get("data_keys") if isinstance(payload, dict) else None
        if not keys and isinstance(data, str):
            try:
                parsed = json.loads(data)
                keys = list(parsed.keys()) if isinstance(parsed, dict) else []
            except json.JSONDecodeError:
                keys = []
        log = {
            "ok": bool(payload.get("ok")) if isinstance(payload, dict) else False,
            "purpose": payload.get("purpose") if isinstance(payload, dict) else None,
            "data_keys": keys or [],
            "follow_extract": payload.get("follow_extract") if isinstance(payload, dict) else None,
        }
        LAST_FOLLOW_LOG_PATH.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
        try:
            with (ENROLLMENT_DIR / "bridge.log").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"event": "submit", **log}) + "\n")
        except OSError:
            pass
        self._send_json({"ok": True})


def write_prove_config(config: Dict[str, Any]) -> Path:
    ENROLLMENT_DIR.mkdir(parents=True, exist_ok=True)
    config = dict(config)
    config.setdefault("recipient", local_recipient())
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return CONFIG_PATH


def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def read_extension_result() -> Optional[Dict[str, Any]]:
    """One-shot pending proof. Do not fall back to last_extension_result
    or every new browser session would inherit the previous user's unlock."""
    return _read_json_file(RESULT_PATH)


def clear_extension_result() -> None:
    if RESULT_PATH.exists():
        RESULT_PATH.unlink()


def ensure_extension_server() -> str:
    global _started
    with _lock:
        if not _started:
            try:
                httpd = ThreadingHTTPServer(("127.0.0.1", PORT), _Handler)
            except OSError:
                _started = True
                return f"http://127.0.0.1:{PORT}/prove.html"
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            _started = True
    return f"http://127.0.0.1:{PORT}/prove.html"
