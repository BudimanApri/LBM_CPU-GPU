// Reproduces the high-Re/U/AoA case from the captured boundary-wiggle report
// and checks that the inlet ramp completes, the sentinel stays clear, and
// far-field density remains quiet after several acoustic crossing times.
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const server = await createServer({
  logLevel: 'error',
  server: { host: '127.0.0.1', port: 5175, strictPort: true },
});
await server.listen();
const browser = await chromium.launch({
  headless: true,
  args: ['--enable-unsafe-webgpu', '--enable-gpu', '--disable-dawn-features=use_dxc'],
});
const page = await browser.newPage({ viewport: { width: 1200, height: 700 } });
const errors: string[] = [];
page.on('console', (message) => {
  if (message.type() === 'error') errors.push(message.text());
});
page.on('pageerror', (error) => errors.push(error.message));

await page.goto('http://127.0.0.1:5175/?resolution=1024x512');
await page.waitForFunction(() => '__lbm' in window, undefined, { timeout: 30_000 });
await page.evaluate(() => {
  const sliders = [...document.querySelectorAll<HTMLInputElement>('input[type=range]')];
  const re = sliders[0]!;
  re.value = String(Math.log10(7079));
  re.dispatchEvent(new Event('input', { bubbles: true }));
  const inlet = sliders[1]!;
  inlet.value = '0.09';
  inlet.dispatchEvent(new Event('input', { bubbles: true }));
  const aoa = sliders.find((slider) => slider.min === '-15' && slider.max === '15');
  if (!aoa) throw new Error('AoA slider not found');
  aoa.value = '8';
  aoa.dispatchEvent(new Event('input', { bubbles: true }));
  const airfoil = [...document.querySelectorAll<HTMLButtonElement>('button')].find(
    (button) => button.textContent === 'Airfoil',
  );
  airfoil?.click();
  const reset = [...document.querySelectorAll<HTMLButtonElement>('button')].find(
    (button) => button.textContent === 'Reset flow',
  );
  reset?.click();
});

await page.waitForTimeout(45_000);
const result = await page.evaluate(async () => {
  const probe = (
    window as unknown as Window & {
      __lbm: {
        nx: number;
        ny: number;
        steps: () => number;
        stability: () => {
          lesEnabled: boolean;
          nanDetected: boolean;
          paused: boolean;
          currentInletU: number;
          inletTargetU: number;
          inletRampProgress: number;
        };
        readMoments: () => Promise<{
          rho: Float32Array;
          ux: Float32Array;
          uy: Float32Array;
        }>;
      };
    }
  ).__lbm;
  const moments = await probe.readMoments();
  let globalMin = Infinity;
  let globalMax = -Infinity;
  let maxSpeed = 0;
  let farSum = 0;
  let farSquared = 0;
  let farCount = 0;
  for (let y = 0; y < probe.ny; y++) {
    for (let x = 0; x < probe.nx; x++) {
      const cell = y * probe.nx + x;
      const rho = moments.rho[cell]!;
      globalMin = Math.min(globalMin, rho);
      globalMax = Math.max(globalMax, rho);
      maxSpeed = Math.max(maxSpeed, Math.hypot(moments.ux[cell]!, moments.uy[cell]!));
      // Undamped, obstacle-free upstream core: any spatial variation here is
      // primarily reflected acoustic energy, not the physical wake.
      if (
        x >= probe.nx * 0.05 &&
        x < probe.nx * 0.2 &&
        y >= probe.ny * 0.12 &&
        y < probe.ny * 0.88
      ) {
        farSum += rho;
        farSquared += rho * rho;
        farCount++;
      }
    }
  }
  const farMean = farSum / farCount;
  const farStd = Math.sqrt(Math.max(0, farSquared / farCount - farMean * farMean));
  return {
    status: document.querySelector('#status')?.textContent ?? '',
    steps: probe.steps(),
    stability: probe.stability(),
    globalMin,
    globalMax,
    maxSpeed,
    farMean,
    farStd,
  };
});

await page.evaluate(() => {
  const view = [...document.querySelectorAll<HTMLSelectElement>('select')].find((select) =>
    [...select.options].some((option) => option.value === 'density'),
  );
  if (view) {
    view.value = 'density';
    view.dispatchEvent(new Event('change', { bubbles: true }));
  }
});
await mkdir('.vitest-attachments', { recursive: true });
await page.screenshot({ path: '.vitest-attachments/acoustic-browser-gate.png' });

await browser.close();
await server.close();

if (errors.length > 0) throw new Error(errors.join('\n'));
if (!result.stability.lesEnabled || result.stability.nanDetected || result.stability.paused) {
  throw new Error(`stability failure: ${JSON.stringify(result.stability)}`);
}
if (result.stability.inletRampProgress < 1200) {
  throw new Error(`inlet ramp incomplete: ${JSON.stringify(result.stability)}`);
}
if (Math.abs(result.stability.currentInletU - 0.09) > 1e-6) {
  throw new Error(`inlet did not reach target: ${JSON.stringify(result.stability)}`);
}
if (![result.globalMin, result.globalMax, result.maxSpeed, result.farStd].every(Number.isFinite)) {
  throw new Error(`non-finite fields: ${JSON.stringify(result)}`);
}
if (result.farStd > 0.005) {
  throw new Error(`far-field acoustic variation remains high: ${JSON.stringify(result)}`);
}

console.log(JSON.stringify(result, null, 2));
