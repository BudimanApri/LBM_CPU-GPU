import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const output = join('C:\\tmp', `lbm-readme-${Date.now()}`);
await mkdir(output, { recursive: true });

const server = await createServer({
  logLevel: 'error',
  server: { host: '127.0.0.1', port: 5174, strictPort: true },
});
await server.listen();
const browser = await chromium.launch({
  headless: true,
  args: ['--enable-unsafe-webgpu', '--enable-gpu', '--disable-dawn-features=use_dxc'],
});
const page = await browser.newPage({ viewport: { width: 1200, height: 700 } });
await page.goto('http://127.0.0.1:5174/');
await page.waitForFunction(() => '__lbm' in window);
await page.waitForFunction(
  () =>
    (
      window as unknown as Window & {
        __lbm: { steps: () => number };
      }
    ).__lbm.steps() > 20_000,
  undefined,
  { timeout: 70_000 },
);

for (let frame = 0; frame < 16; frame++) {
  await page.screenshot({ path: join(output, `frame-${String(frame).padStart(2, '0')}.png`) });
  await page.waitForTimeout(350);
}

await browser.close();
await server.close();
console.log(output);
