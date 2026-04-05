import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Use relative paths so the build works when served from any base URL
  // (Streamlit static file serving, GitHub Pages, local file://, etc.)
  base: "./",
  server: {
    host: "127.0.0.1",
    port: 4173,
  },
  build: {
    outDir: "dist",
    assetsDir: "assets",
  },
});
