import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Honour PORT so a second checkout can run alongside one already on 5173.
    port: Number(process.env.PORT) || 5173,
    // Forward API calls to the iMATCH backend so local dev needs no CORS grant
    // and the browser never sees a cross-origin biometric request.
    proxy: {
      '/api': {
        target: 'http://localhost:8443',
        changeOrigin: true,
      },
    },
  },
})
