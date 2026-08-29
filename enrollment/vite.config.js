import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  define: {
    global: "globalThis",
  },
  build: {
    lib: {
      entry: resolve(__dirname, "src/sdk-entry.js"),
      name: "VoiceSenseZKTLS",
      formats: ["iife"],
      fileName: () => "zktls-bundle.js",
    },
    outDir: resolve(__dirname, "../app/primus_enroll/frontend/vendor"),
    emptyOutDir: false,
    sourcemap: false,
  },
});
