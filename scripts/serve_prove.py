"""Keep the Primus prove.html server on 127.0.0.1:8765 independent of Streamlit."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.enrollment import local_recipient, primus_settings
from src.extension_bridge import ensure_extension_server, write_prove_config


def main() -> None:
    ps = primus_settings()
    handle = ps.get("x_follow_handle") or "its_perseus_1"
    write_prove_config(
        {
            "app_id": ps["app_id"],
            "app_secret": ps["app_secret"],
            "recipient": local_recipient(),
            "att_mode": ps["att_mode"],
            "addition_params": {},
            "return_url": "http://127.0.0.1:8502/?verified=1",
            "flows": {
                "x_follow": {
                    "template_id": ps["x_follow_template_id"],
                    "field": ps["x_follow_field"],
                    "op": ps["x_follow_op"],
                    "value": ps["x_follow_value"],
                    "label": f"Prove you follow @{handle}",
                    "use_conditions": False,
                },
            },
        }
    )
    url = ensure_extension_server()
    print(url, flush=True)
    while True:
        time.sleep(30)


if __name__ == "__main__":
    main()
