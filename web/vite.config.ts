import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 3000,
    proxy: {
      '/iris3d.v1': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
});
