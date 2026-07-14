import { describe, expect, it } from 'vitest';
import {
  BRUSH_PARAMS_BYTE_SIZE,
  BRUSH_PARAMS_OFFSETS,
  FLAG_DYE_ENABLED,
  FLAG_PERIODIC_Y,
  PARAMS_BYTE_SIZE,
  PARAMS_OFFSETS,
  VIEW_MODE_MASK,
  VIEW_MODE_SHIFT,
  encodeBrushParams,
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
      substeps: 24,
    });
  });

  it('encodeParams writes each field at its offset, little-endian', () => {
    const buf = encodeParams({
      nx: 128,
      ny: 64,
      tau: 0.65,
      inletU: 0.08,
      periodicY: true,
      dyeEnabled: true,
      viewMode: 'vorticity',
      stepIndex: 7,
      substeps: 5,
    });
    expect(buf.byteLength).toBe(PARAMS_BYTE_SIZE);
    const dv = new DataView(buf);
    expect(dv.getUint32(PARAMS_OFFSETS.nx, true)).toBe(128);
    expect(dv.getUint32(PARAMS_OFFSETS.ny, true)).toBe(64);
    expect(dv.getFloat32(PARAMS_OFFSETS.tau, true)).toBe(Math.fround(0.65));
    expect(dv.getFloat32(PARAMS_OFFSETS.inletU, true)).toBe(Math.fround(0.08));
    const flags = dv.getUint32(PARAMS_OFFSETS.flags, true);
    expect(flags & FLAG_PERIODIC_Y).toBe(FLAG_PERIODIC_Y);
    expect(flags & FLAG_DYE_ENABLED).toBe(FLAG_DYE_ENABLED);
    expect((flags >> VIEW_MODE_SHIFT) & VIEW_MODE_MASK).toBe(1); // vorticity = 1
    expect(dv.getUint32(PARAMS_OFFSETS.stepIndex, true)).toBe(7);
    expect(dv.getUint32(PARAMS_OFFSETS.substeps, true)).toBe(5);
    // Trailing padding stays zeroed.
    expect(dv.getUint32(28, true)).toBe(0);
  });

  it('clears the periodic/dye flags for free-slip, dye-off, velocity view', () => {
    const dv = new DataView(
      encodeParams({
        nx: 4,
        ny: 4,
        tau: 0.6,
        inletU: 0.05,
        periodicY: false,
        dyeEnabled: false,
        viewMode: 'velocity',
        stepIndex: 0,
        substeps: 1,
      }),
    );
    expect(dv.getUint32(PARAMS_OFFSETS.flags, true)).toBe(0);
  });

  it.each([
    ['velocity', 0],
    ['vorticity', 1],
    ['density', 2],
    ['dye', 3],
  ] as const)('packs view mode %s as code %d', (viewMode, code) => {
    const dv = new DataView(
      encodeParams({
        nx: 4,
        ny: 4,
        tau: 0.6,
        inletU: 0.05,
        periodicY: false,
        dyeEnabled: false,
        viewMode,
        stepIndex: 0,
        substeps: 1,
      }),
    );
    const flags = dv.getUint32(PARAMS_OFFSETS.flags, true);
    expect((flags >> VIEW_MODE_SHIFT) & VIEW_MODE_MASK).toBe(code);
  });
});

describe('BrushParams uniform layout', () => {
  it('leading vec2f keeps its 8-byte alignment and the struct is 16-padded', () => {
    expect(BRUSH_PARAMS_BYTE_SIZE % 16).toBe(0);
    expect(BRUSH_PARAMS_OFFSETS).toEqual({
      centerX: 0,
      centerY: 4,
      radius: 8,
      mode: 12,
      nx: 16,
      ny: 20,
    });
  });

  it('encodeBrushParams writes each field at its offset', () => {
    const buf = encodeBrushParams({
      centerX: 100.5,
      centerY: 60.25,
      radius: 7,
      paint: true,
      nx: 512,
      ny: 256,
    });
    expect(buf.byteLength).toBe(BRUSH_PARAMS_BYTE_SIZE);
    const dv = new DataView(buf);
    expect(dv.getFloat32(BRUSH_PARAMS_OFFSETS.centerX, true)).toBe(Math.fround(100.5));
    expect(dv.getFloat32(BRUSH_PARAMS_OFFSETS.centerY, true)).toBe(Math.fround(60.25));
    expect(dv.getFloat32(BRUSH_PARAMS_OFFSETS.radius, true)).toBe(7);
    expect(dv.getUint32(BRUSH_PARAMS_OFFSETS.mode, true)).toBe(1);
    expect(dv.getUint32(BRUSH_PARAMS_OFFSETS.nx, true)).toBe(512);
    expect(dv.getUint32(BRUSH_PARAMS_OFFSETS.ny, true)).toBe(256);
    const erase = new DataView(
      encodeBrushParams({ centerX: 0, centerY: 0, radius: 1, paint: false, nx: 4, ny: 4 }),
    );
    expect(erase.getUint32(BRUSH_PARAMS_OFFSETS.mode, true)).toBe(0);
  });
});
