/* VoiceSense Primus enrollment UI. Voice never enters this iframe. */
(function () {
  const root = document.getElementById("root");
  let args = {};
  let busy = false;

  function resize() {
    const h = Math.max(document.body.scrollHeight, 280);
    window.Streamlit.setFrameHeight(h + 16);
  }

  function el(html) {
    root.innerHTML = html;
    resize();
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function conditionsFromArgs() {
    const field = args.att_field || "screen_name";
    const op = args.att_op || "SHA256";
    const value = args.att_value || "";
    const item = { field: field, op: op };
    if (op !== "SHA256" && op !== "SHA256_EX" && value !== "") {
      item.value = value;
    }
    return [[item]];
  }

  function renderIdle(message, kind) {
    const sdkOk = !!(window.PrimusZKTLS || (args.sdk_present && window.PrimusZKTLS));
    const bundlePresent = args.sdk_present !== false && typeof args.sdk_present !== "undefined"
      ? args.sdk_present
      : !!document.querySelector("script[src*='zktls-bundle']");
    const hasSdk = typeof window.PrimusZKTLS === "function";
    const label = escapeHtml(args.button_label || "Prove with Primus");
    const purpose = escapeHtml(args.purpose || "uniqueness");
    const status = message
      ? `<div class="msg ${kind || "info"}">${escapeHtml(message)}</div>`
      : "";

    let sdkNote = "";
    if (!hasSdk) {
      sdkNote = `
        <div class="msg warn">
          Primus JS SDK bundle is not built yet. In a terminal run
          <code>cd enrollment && npm install && npm run build</code>,
          then restart Streamlit. You can still paste an attestation JSON
          in the app above.
        </div>`;
    }

    el(`
      <div class="card">
        <div class="kicker">Primus zkTLS · ${purpose}</div>
        <p class="muted">
          This step proves a Web2 account or eligibility condition.
          It does <b>not</b> receive your voice recording.
          Identity fields are requested as SHA-256 or true/false only.
        </p>
        ${sdkNote}
        ${status}
        <button id="prove" ${hasSdk && args.app_id && args.template_id ? "" : "disabled"}>
          ${label}
        </button>
        <p class="hint">Requires the Primus browser extension and a Developer Hub template.</p>
      </div>
    `);

    const btn = document.getElementById("prove");
    if (btn) btn.addEventListener("click", onProve);
    resize();
  }

  async function onProve() {
    if (busy) return;
    if (typeof window.PrimusZKTLS !== "function") {
      renderIdle("SDK bundle missing.", "warn");
      return;
    }
    busy = true;
    renderIdle("Starting Primus attestation… keep the extension popup in view.", "info");
    try {
      const zk = new window.PrimusZKTLS();
      const appId = args.app_id;
      const appSecret = args.app_secret || "";
      if (appSecret) {
        await zk.init(appId, appSecret);
      } else {
        await zk.init(appId);
      }

      const request = zk.generateRequestParams(args.template_id, args.recipient);
      const mode = args.att_mode === "mpctls" ? "mpctls" : "proxytls";
      request.setAttMode({ algorithmType: mode });
      request.setAttConditions(conditionsFromArgs());
      let extra = {
        voicesense: "enrollment-only",
        purpose: args.purpose || "uniqueness",
        no_audio: true,
      };
      if (args.addition_params) {
        try {
          const parsed = JSON.parse(args.addition_params);
          if (parsed && typeof parsed === "object") {
            extra = Object.assign(extra, parsed);
          }
        } catch (e) {
          /* keep defaults */
        }
      }
      request.setAdditionParams(JSON.stringify(extra));

      const requestStr = request.toJsonString();
      let signed = requestStr;
      if (typeof zk.sign === "function" && appSecret) {
        signed = await zk.sign(requestStr);
      } else if (!appSecret) {
        throw new Error(
          "PRIMUS_APP_SECRET is not set. For local research use, add it to .env " +
            "(test-mode signing). Production apps must sign on a server instead."
        );
      }

      const attestation = await zk.startAttestation(signed);
      const verifyResult = await zk.verifyAttestation(attestation);
      if (verifyResult !== true) {
        throw new Error("Primus signature verification returned false.");
      }

      window.Streamlit.setComponentValue({
        ok: true,
        purpose: args.purpose || "uniqueness",
        sdk_verified: true,
        template_id: args.template_id,
        attestation: attestation,
      });
      renderIdle("Attestation verified. Fingerprint saved in the app — no voice involved.", "ok");
    } catch (err) {
      const msg = (err && (err.message || err.code)) ? String(err.message || err.code) : String(err);
      window.Streamlit.setComponentValue({
        ok: false,
        error: msg,
      });
      renderIdle(msg, "err");
    } finally {
      busy = false;
      resize();
    }
  }

  window.addEventListener("message", function (event) {
    const data = event.data || {};
    if (data.type !== "streamlit:render") return;
    args = data.args || {};
    if (!busy) renderIdle();
  });

  window.Streamlit.setComponentReady();
  window.Streamlit.setFrameHeight(320);
})();
