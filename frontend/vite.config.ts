import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the UI runs on Vite (5173) and proxies the API (8000), so it is still
// same-origin from the browser's view. In production FastAPI serves the built files.
const api = "http://127.0.0.1:8000";
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: Object.fromEntries(
      ["/v1", "/health", "/ready", "/metrics"].map((p) => [p, { target: api }]),
    ),
  },
  build: { sourcemap: true },
});
