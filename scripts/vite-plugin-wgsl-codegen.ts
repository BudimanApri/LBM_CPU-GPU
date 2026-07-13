import type { Plugin } from 'vite';
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { generateD2Q9Wgsl } from './wgsl-codegen.ts';

const OUT = 'src/gpu/shaders/generated/d2q9-constants.wgsl';
const WATCHED = 'src/solver/constants.ts';

export function wgslCodegenPlugin(): Plugin {
  function regenerate(): void {
    const outPath = resolve(OUT);
    mkdirSync(dirname(outPath), { recursive: true });
    writeFileSync(outPath, generateD2Q9Wgsl(), 'utf8');
  }

  return {
    name: 'wgsl-codegen',
    buildStart() {
      regenerate();
    },
    configureServer(server) {
      const watchedPath = resolve(WATCHED);
      server.watcher.add(watchedPath);
      server.watcher.on('change', (file) => {
        if (file === watchedPath) {
          regenerate();
          server.ws.send({ type: 'full-reload' });
        }
      });
    },
  };
}
