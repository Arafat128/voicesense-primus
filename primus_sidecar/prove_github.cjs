/**
 * Optional backend zkTLS of a PUBLIC GitHub API URL.
 * Does not touch voice recordings or joblib files.
 *
 *   cd primus_sidecar
 *   npm install
 *   node prove_github.cjs
 *
 * Requires PRIMUS_APP_ID + PRIMUS_APP_SECRET in ../.env
 */
const fs = require("fs");
const path = require("path");

function loadEnv() {
  const envPath = path.join(__dirname, "..", ".env");
  if (!fs.existsSync(envPath)) return;
  for (const line of fs.readFileSync(envPath, "utf8").split(/\r?\n/)) {
    const m = line.match(/^([A-Z0-9_]+)=(.*)$/);
    if (m && !process.env[m[1]]) process.env[m[1]] = m[2];
  }
}

async function main() {
  loadEnv();
  const appId = process.env.PRIMUS_APP_ID;
  const appSecret = process.env.PRIMUS_APP_SECRET;
  if (!appId || !appSecret) {
    console.error("Set PRIMUS_APP_ID and PRIMUS_APP_SECRET in ../.env");
    process.exit(1);
  }
  let PrimusCoreTLS;
  try {
    ({ PrimusCoreTLS } = require("@primuslabs/zktls-core-sdk"));
  } catch (err) {
    console.error("Run: cd primus_sidecar && npm install");
    throw err;
  }
  const zkTLS = new PrimusCoreTLS();
  await zkTLS.init(appId, appSecret);
  const request = {
    url: "https://api.github.com/repos/Arafat128/VoiceSense-PD",
    method: "GET",
    header: { "User-Agent": "VoiceSense-PD-primus-sidecar" },
    body: "",
  };
  const responseResolves = [
    { keyName: "full_name", parsePath: "$.full_name" },
  ];
  const generateRequest = zkTLS.generateRequestParams(request, responseResolves);
  generateRequest.setAttMode({ algorithmType: process.env.PRIMUS_ATT_MODE || "proxytls" });
  generateRequest.setAttConditions([{ field: "full_name", op: "SHA256" }]);
  const attestation = await zkTLS.startAttestation(generateRequest);
  const ok = zkTLS.verifyAttestation(attestation);
  console.log(JSON.stringify({ verified: ok, attestation }, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
