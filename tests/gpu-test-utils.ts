// Shared GPU test helpers. Not itself a test file (doesn't match the
// tests/gpu-*.test.ts glob).
import { expect } from 'vitest';

/**
 * WebGPU shader compile errors don't throw synchronously from
 * createShaderModule/createRenderPipeline/createComputePipeline -- an
 * invalid module silently produces a pipeline whose draws/dispatches are
 * dropped, which reads as "everything is zero" rather than a clear error
 * (this cost real debugging time once already in this project). Call this
 * right after creating any shader module in a test to fail fast instead.
 */
export async function assertShaderCompiles(module: GPUShaderModule): Promise<void> {
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  expect(errors, errors.map((e) => `${e.lineNum}:${e.linePos}: ${e.message}`).join('\n')).toEqual(
    [],
  );
}
