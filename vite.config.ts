import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';
import { wgslCodegenPlugin } from './scripts/vite-plugin-wgsl-codegen.ts';

export default defineConfig({
  plugins: [wgslCodegenPlugin()],
  test: {
    projects: [
      {
        extends: true,
        test: {
          name: 'node',
          include: ['tests/**/*.test.ts'],
          exclude: ['tests/gpu-parity.test.ts'],
          // The CPU-solver validation tests march thousands of LBM steps to
          // steady state; Vitest workers run them far slower than plain Node.
          testTimeout: 60_000,
        },
      },
      {
        extends: true,
        test: {
          name: 'browser',
          include: ['tests/gpu-parity.test.ts'],
          testTimeout: 120_000,
          browser: {
            enabled: true,
            headless: true,
            provider: playwright({
              launchOptions: {
                // The headless shell exposes the real D3D12 adapter with
                // --enable-unsafe-webgpu, but ships no DXC DLLs
                // (dxcompiler/dxil); disabling Dawn's use_dxc falls back to
                // FXC (D3DCompiler_47.dll), which Windows itself provides.
                args: ['--enable-unsafe-webgpu', '--enable-gpu', '--disable-dawn-features=use_dxc'],
              },
            }),
            instances: [{ browser: 'chromium' }],
          },
        },
      },
    ],
  },
});
