import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { generateD2Q9Wgsl } from './wgsl-codegen.ts';

const outPath = 'src/gpu/shaders/generated/d2q9-constants.wgsl';
mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, generateD2Q9Wgsl(), 'utf8');
console.log(`generated ${outPath}`);
