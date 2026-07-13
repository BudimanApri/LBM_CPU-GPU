import { CX, CY, WEIGHTS, OPPOSITE, Q } from '../src/solver/constants.ts';

function i32Array(name: string, values: readonly number[]): string {
  return `const ${name}: array<i32, ${values.length}> = array<i32, ${values.length}>(${values.join(', ')});`;
}

function f32Array(name: string, values: readonly number[]): string {
  const literals = values.map((v) => String(v));
  return `const ${name}: array<f32, ${values.length}> = array<f32, ${values.length}>(${literals.join(', ')});`;
}

export function generateD2Q9Wgsl(): string {
  return [
    '// AUTO-GENERATED from src/solver/constants.ts by scripts/wgsl-codegen.ts. DO NOT EDIT.',
    `const D2Q9_Q: i32 = ${Q};`,
    i32Array('D2Q9_CX', CX),
    i32Array('D2Q9_CY', CY),
    f32Array('D2Q9_W', WEIGHTS),
    i32Array('D2Q9_OPP', OPPOSITE),
    '',
  ].join('\n');
}
