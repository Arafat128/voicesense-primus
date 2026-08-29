# VoiceSense Primus enrollment bundle

This folder only builds the browser SDK used by the **optional** uniqueness / age
proof in the Streamlit app. It never sees voice recordings.

## One-time build

```bash
cd enrollment
npm install
npm run build
```

That writes `app/primus_enroll/frontend/vendor/zktls-bundle.js`.

Restart Streamlit afterwards.

## Developer Hub

1. Create a project at https://dev.primuslabs.xyz and copy `appID` / `appSecret`.
2. Create or pick a **GitHub** or **X** data template (account ownership).
3. Put the IDs in the repo-root `.env` (see `.env.example`).
4. Install the [Primus extension](https://chromewebstore.google.com/detail/primus/oeiomhmbaapihbilkfkhmlajkeegnjhe).
5. In VoiceSense open **Privacy & enrollment** → **Prove uniqueness**.

Use SHA-256 (the app already requests `op: SHA256`) so VoiceSense stores a
fingerprint, not your handle.

`PRIMUS_APP_SECRET` is used for **local test-mode signing only**. Do not deploy
this Streamlit app to the public internet with the secret in `.env`.
