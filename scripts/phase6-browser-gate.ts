// Phase 6 browser gate. It starts an internal Vite server and validates the
// live frame loop, high-Re LES case, and 2048x1024 allocation using
// Chromium's real WebGPU path.
import { chromium, type Page } from 'playwright';
import { createServer } from 'vite';

interface LiveSnapshot {
  status: string;
  nx: number;
  ny: number;
  workgroup: string;
  steps: number;
  stability: {
    lesEnabled: boolean;
    nanDetected: boolean;
    gpuDrainMs: number | null;
    paused: boolean;
  };
}

async function waitForSimulation(page: Page): Promise<void> {
  await page.waitForFunction(() => '__lbm' in window, undefined, { timeout: 30_000 });
}

async function snapshot(page: Page): Promise<LiveSnapshot> {
  return page.evaluate(() => {
    const probe = (
      window as unknown as Window & {
        __lbm: {
          nx: number;
          ny: number;
          workgroup: string;
          steps: () => number;
          stability: () => LiveSnapshot['stability'];
          setKForValidation: (k: number) => void;
          resumeAdaptiveK: () => void;
        };
      }
    ).__lbm;
    return {
      status: document.querySelector('#status')?.textContent ?? '',
      nx: probe.nx,
      ny: probe.ny,
      workgroup: probe.workgroup,
      steps: probe.steps(),
      stability: probe.stability(),
    };
  });
}

async function setHighReAirfoil(page: Page): Promise<void> {
  await page.evaluate(() => {
    (
      window as unknown as Window & { __lbm: { resumeAdaptiveK: () => void } }
    ).__lbm.resumeAdaptiveK();
    const sliders = [...document.querySelectorAll<HTMLInputElement>('input[type=range]')];
    const re = sliders[0]!;
    re.value = String(Math.log10(5000));
    re.dispatchEvent(new Event('input', { bubbles: true }));
    const aoa = sliders.find((slider) => slider.min === '-15' && slider.max === '15');
    if (!aoa) throw new Error('AoA slider not found');
    aoa.value = '10';
    aoa.dispatchEvent(new Event('input', { bubbles: true }));
    const airfoil = [...document.querySelectorAll<HTMLButtonElement>('button')].find(
      (button) => button.textContent === 'Airfoil',
    );
    airfoil?.click();
  });
}

async function momentBounds(
  page: Page,
): Promise<{ rhoMin: number; rhoMax: number; maxSpeed: number }> {
  return page.evaluate(async () => {
    const probe = (
      window as unknown as Window & {
        __lbm: {
          readMoments: () => Promise<{
            rho: Float32Array;
            ux: Float32Array;
            uy: Float32Array;
          }>;
        };
      }
    ).__lbm;
    const moments = await probe.readMoments();
    let rhoMin = Infinity;
    let rhoMax = -Infinity;
    let maxSpeed = 0;
    for (let i = 0; i < moments.rho.length; i++) {
      const rho = moments.rho[i]!;
      const speed = Math.hypot(moments.ux[i]!, moments.uy[i]!);
      if (!Number.isFinite(rho) || !Number.isFinite(speed)) {
        return { rhoMin: NaN, rhoMax: NaN, maxSpeed: NaN };
      }
      rhoMin = Math.min(rhoMin, rho);
      rhoMax = Math.max(rhoMax, rho);
      maxSpeed = Math.max(maxSpeed, speed);
    }
    return { rhoMin, rhoMax, maxSpeed };
  });
}

const server = await createServer({
  logLevel: 'error',
  server: { host: '127.0.0.1', port: 5173, strictPort: true },
});
await server.listen();

const browser = await chromium.launch({
  headless: true,
  args: ['--enable-unsafe-webgpu', '--enable-gpu', '--disable-dawn-features=use_dxc'],
});
const page = await browser.newPage();
const errors: string[] = [];
page.on('console', (message) => {
  if (message.type() === 'error') errors.push(message.text());
});
page.on('pageerror', (error) => errors.push(error.message));

await page.goto('http://127.0.0.1:5173/?resolution=1024x512');
await waitForSimulation(page);
await page.evaluate(() => {
  (
    window as unknown as Window & { __lbm: { setKForValidation: (k: number) => void } }
  ).__lbm.setKForValidation(3);
});
await page.waitForTimeout(8_000);
const performance = await snapshot(page);

await setHighReAirfoil(page);
await page.waitForTimeout(30_000);
const highRe = await snapshot(page);
const bounds = await momentBounds(page);

await page.goto('http://127.0.0.1:5173/?resolution=2048x1024');
await waitForSimulation(page);
await page.waitForTimeout(8_000);
const highResolution = await snapshot(page);

await browser.close();
await server.close();

if (!/fps\s+(?:[6-9]\d|\d{3,})/.test(performance.status)) {
  throw new Error(`1024x512 performance gate failed:\n${performance.status}`);
}
if (!highRe.stability.lesEnabled || highRe.stability.nanDetected || highRe.stability.paused) {
  throw new Error(`Re=5000 LES stability gate failed: ${JSON.stringify(highRe.stability)}`);
}
if (![bounds.rhoMin, bounds.rhoMax, bounds.maxSpeed].every(Number.isFinite)) {
  throw new Error(`Re=5000 fields contain non-finite values: ${JSON.stringify(bounds)}`);
}
if (highResolution.nx !== 2048 || highResolution.ny !== 1024) {
  throw new Error(`2048x1024 allocation gate failed: ${JSON.stringify(highResolution)}`);
}
if (errors.length > 0) throw new Error(`Browser errors:\n${errors.join('\n')}`);

console.log(JSON.stringify({ performance, highRe, bounds, highResolution }, null, 2));
