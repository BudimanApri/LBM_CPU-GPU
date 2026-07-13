import { describe, expect, it } from 'vitest';
import { generateD2Q9Wgsl } from '../scripts/wgsl-codegen.ts';

describe('generateD2Q9Wgsl', () => {
  const wgsl = generateD2Q9Wgsl();

  it('declares the scalar count constant', () => {
    expect(wgsl).toMatch(/const D2Q9_Q: i32 = 9;/);
  });

  it.each([
    ['D2Q9_CX', 'i32'],
    ['D2Q9_CY', 'i32'],
    ['D2Q9_OPP', 'i32'],
    ['D2Q9_SPEC', 'i32'],
    ['D2Q9_W', 'f32'],
  ])('declares %s as a well-formed array<%s, 9> with 9 elements', (name, type) => {
    const re = new RegExp(`const ${name}: array<${type}, 9> = array<${type}, 9>\\(([^)]+)\\);`);
    const match = re.exec(wgsl);
    expect(match).not.toBeNull();
    const elements = match![1]!.split(',').map((s) => s.trim());
    expect(elements).toHaveLength(9);
    for (const el of elements) {
      expect(el.length).toBeGreaterThan(0);
    }
  });

  it('is deterministic (same input, same output)', () => {
    expect(generateD2Q9Wgsl()).toBe(wgsl);
  });
});
