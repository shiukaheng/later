import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const backend = process.env.LATER_WEB_ORIGIN ?? process.env.LARS_WEB_ORIGIN ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  build: { outDir: '../src/later/web_dist', emptyOutDir: true },
  server: { proxy: Object.fromEntries(['/next', '/rate', '/api'].map(path => [path, backend])) },
})
