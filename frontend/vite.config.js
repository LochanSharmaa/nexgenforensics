import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // PORT lets tooling assign a free port when 5173 is already taken.
    port: Number(process.env.PORT) || 5173,
    // Forward API calls to the iMATCH backend so local dev needs no CORS grant
    // and the browser never sees a cross-origin biometric request.
    proxy: {
      '/api': {
        // IMATCH_PROXY_TARGET lets a second dev instance point at a backend on
        // a non-default port (e.g. when 8443 is already held by another
        // session). Default unchanged.
        target: process.env.IMATCH_PROXY_TARGET || 'http://localhost:8443',
        changeOrigin: true,
      },
      // Image Intelligence Engine (public-web provenance). Proxied for the same
      // reason as /api: same-origin in dev means no CORS grant per Vite port,
      // and no cross-origin request carrying an investigator's bearer token.
      // The rewrite strips the prefix so IIE still sees its own /api/v1 paths.
      '/iie': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/iie/, ''),
      },
    },
  },
})
