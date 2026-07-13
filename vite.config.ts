import { defineConfig } from 'vitest/config';
import { wgslCodegenPlugin } from './scripts/vite-plugin-wgsl-codegen.ts';

export default defineConfig({
  plugins: [wgslCodegenPlugin()],
  test: {
    include: ['tests/**/*.test.ts'],
  },
});
