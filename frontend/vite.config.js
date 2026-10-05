import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server runs on 5173 by default, matching the backend's CORS allowance
// (see backend/api/main.py — CORS is wide open ("*") outside production).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
});
