import { defineConfig } from 'vitest/config';
import { wgslCodegenPlugin } from './scripts/vite-plugin-wgsl-codegen.ts';

export default defineConfig({
  plugins: [wgslCodegenPlugin()],
  test: {
    include: ['tests/**/*.test.ts'],
    // The CPU-solver validation tests march thousands of LBM steps to steady
    // state; Vitest's worker execution runs them far slower than plain Node.
    testTimeout: 60_000,
  },
});
