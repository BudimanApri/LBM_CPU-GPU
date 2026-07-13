import { describe, expect, it } from 'vitest';
import {
  FLAG_PERIODIC_Y,
  PARAMS_BYTE_SIZE,
  PARAMS_OFFSETS,
  encodeParams,
} from '../src/gpu/buffers.ts';

// CLAUDE.md gotcha #5: WGSL alignment rules can silently corrupt the params
// struct. This test pins the TS-side encoding to the documented byte offsets
// of the WGSL `Params` struct in lbm.wgsl. Whenever a field is added there,
// extend PARAMS_OFFSETS, encodeParams, and this test in the same commit.
describe('Params uniform layout', () => {
  it('is padded to a 16-byte multiple', () => {
    expect(PARAMS_BYTE_SIZE % 16).toBe(0);
  });

  it('field offsets match the WGSL struct (scalars, 4-byte aligned, in order)', () => {
    expect(PARAMS_OFFSETS).toEqual({
      nx: 0,
      ny: 4,
      tau: 8,
      inletU: 12,
      flags: 16,
      stepIndex: 20,
    });
  });

  it('encodeParams writes each field at its offset, little-endian', () => {
    const buf = encodeParams({
      nx: 128,
      ny: 64,
      tau: 0.65,
      inletU: 0.08,
      periodicY: true,
      stepIndex: 7,
    });
    expect(buf.byteLength).toBe(PARAMS_BYTE_SIZE);
    const dv = new DataView(buf);
    expect(dv.getUint32(PARAMS_OFFSETS.nx, true)).toBe(128);
    expect(dv.getUint32(PARAMS_OFFSETS.ny, true)).toBe(64);
    expect(dv.getFloat32(PARAMS_OFFSETS.tau, true)).toBe(Math.fround(0.65));
    expect(dv.getFloat32(PARAMS_OFFSETS.inletU, true)).toBe(Math.fround(0.08));
    expect(dv.getUint32(PARAMS_OFFSETS.flags, true)).toBe(FLAG_PERIODIC_Y);
    expect(dv.getUint32(PARAMS_OFFSETS.stepIndex, true)).toBe(7);
    // Trailing padding stays zeroed.
    expect(dv.getUint32(24, true)).toBe(0);
    expect(dv.getUint32(28, true)).toBe(0);
  });

  it('clears the periodic flag for free-slip walls', () => {
    const dv = new DataView(
      encodeParams({ nx: 4, ny: 4, tau: 0.6, inletU: 0.05, periodicY: false, stepIndex: 0 }),
    );
    expect(dv.getUint32(PARAMS_OFFSETS.flags, true)).toBe(0);
  });
});
