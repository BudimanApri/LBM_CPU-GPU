import { describe, expect, it } from 'vitest';
import { distributionBufferBytes, supportsLbmResolution } from '../src/gpu/buffers.ts';
import {
  DEFAULT_RESOLUTION,
  requestedResolution,
  resolutionDimensions,
} from '../src/gpu/resolution.ts';

function deviceWithLimits(storage: number, buffer: number): GPUDevice {
  return {
    limits: { maxStorageBufferBindingSize: storage, maxBufferSize: buffer },
  } as unknown as GPUDevice;
}

describe('lattice resolution guards', () => {
  it('parses supported query values and falls back for invalid input', () => {
    expect(requestedResolution('?resolution=2048x1024')).toBe('2048x1024');
    expect(requestedResolution('?resolution=banana')).toBe(DEFAULT_RESOLUTION);
    expect(resolutionDimensions('512x256')).toEqual({ nx: 512, ny: 256 });
  });

  it('checks the full 9-population binding against both WebGPU limits', () => {
    const bytes = distributionBufferBytes(2048, 1024);
    expect(bytes).toBe(75_497_472);
    expect(supportsLbmResolution(deviceWithLimits(bytes, bytes), 2048, 1024)).toBe(true);
    expect(supportsLbmResolution(deviceWithLimits(bytes - 4, bytes), 2048, 1024)).toBe(false);
    expect(supportsLbmResolution(deviceWithLimits(bytes, bytes - 4), 2048, 1024)).toBe(false);
  });
});
