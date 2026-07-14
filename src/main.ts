import './style.css';
import { initGPU, observeCanvasResize } from './gpu/context.ts';
import {
  createLbmBuffers,
  writeBrushParams,
  writeMask,
  writeParams,
  writeUniformEquilibrium,
  type LbmBuffers,
} from './gpu/buffers.ts';
import { createBrushPipelines, createLbmPipeline, createRenderPipeline } from './gpu/pipelines.ts';
import { clampU, solveTauForRe, type TauSolution } from './solver/units.ts';
import {
  backwardStepPreset,
  cylinderPreset,
  nacaPreset,
  plateInclinedPreset,
  plateNormalPreset,
  type PresetResult,
} from './geometry/presets.ts';
import { buildControls, type ControlsState, type PresetKind } from './ui/controls.ts';

// Lattice resolution. Fixed for Phase 3; the 512x256 / 1024x512 / 2048x1024
// selector arrives with the performance work (Phases 4/6). Keep the CSS
// aspect-ratio of #gpu-canvas in sync (2 / 1).
const NX = 512;
const NY = 256;
/** Solver substeps per rendered frame (adaptive K comes in Phase 4/6). */
const K_SUBSTEPS = 3;

const PLATE_HEIGHT = 48;
const PLATE_LENGTH = 64;
const PLATE_ANGLE_DEG = 30;
const STEP_HEIGHT = Math.round(NY / 3);

function showFallback(): void {
  document.getElementById('webgpu-fallback')?.classList.remove('hidden');
  document.getElementById('stage')?.remove();
  document.getElementById('panel')?.remove();
}

interface PointerStroke {
  points: { x: number; y: number }[];
  erase: boolean;
}

async function main(): Promise<void> {
  const canvasEl = document.getElementById('gpu-canvas');
  const panelEl = document.getElementById('panel');
  if (!(canvasEl instanceof HTMLCanvasElement) || !(panelEl instanceof HTMLElement)) {
    showFallback();
    return;
  }
  // Narrowed rebindings so closures below see the non-null types.
  const canvas = canvasEl;
  const panel = panelEl;
  const gpu = await initGPU(canvas);
  if (!gpu) {
    showFallback();
    return;
  }
  const { device, context, format } = gpu;
  observeCanvasResize(canvas, device);

  const buffers: LbmBuffers = createLbmBuffers(device, NX, NY);
  const lbm = createLbmPipeline(device, buffers);
  const brush = createBrushPipelines(device, buffers);
  const render = createRenderPipeline(device, buffers, format);

  // --- simulation state ---
  const state: ControlsState = {
    re: 100,
    u: 0.05,
    periodicY: false,
    brushRadius: 6,
    brushErase: false,
    cylinderDiameter: 24,
    nacaDigits: '4412',
    nacaAlphaDeg: 0,
  };
  let stepCount = 0; // parity: f[stepCount % 2] holds the current state
  let paused = false;
  let stepOnce = false;
  let currentD = state.cylinderDiameter;
  let lastPreset: PresetKind = 'cylinder';
  let tauSolution: TauSolution = solveTauForRe(state.re, state.u, currentD);
  // rAF-coalesced edit intents (never more than one preset rasterization
  // and one diff/reconcile chain per frame, however fast sliders fire).
  let presetDirty: PresetKind | null = null;
  let maskEdited = false;
  const stroke: PointerStroke = { points: [], erase: false };
  let lastStamp: { x: number; y: number } | null = null;

  function applyFlowParams(): void {
    state.u = clampU(state.u);
    tauSolution = solveTauForRe(state.re, state.u, currentD);
    writeParams(device, buffers, {
      nx: NX,
      ny: NY,
      tau: tauSolution.tau,
      inletU: state.u,
      periodicY: state.periodicY,
      stepIndex: 0,
    });
  }

  function buildPreset(kind: PresetKind): PresetResult {
    switch (kind) {
      case 'cylinder':
        return cylinderPreset(NX, NY, state.cylinderDiameter);
      case 'airfoil':
        return nacaPreset(NX, NY, state.nacaDigits, Math.round(NX * 0.19), state.nacaAlphaDeg);
      case 'plate-normal':
        return plateNormalPreset(NX, NY, PLATE_HEIGHT);
      case 'plate-inclined':
        return plateInclinedPreset(NX, NY, PLATE_LENGTH, PLATE_ANGLE_DEG);
      case 'step':
        return backwardStepPreset(NX, NY, STEP_HEIGHT);
    }
  }

  function resetFlow(): void {
    writeUniformEquilibrium(device, buffers, (stepCount % 2) as 0 | 1, 1, state.u, 0);
  }

  // --- initial scene: cylinder at Re=100, flow already running ---
  {
    const preset = buildPreset('cylinder');
    currentD = preset.d;
    writeMask(device, buffers, preset.mask, true);
    applyFlowParams();
    resetFlow();
  }

  // --- controls ---
  const controls = buildControls(panel, state, {
    onFlowParamsChange: () => {
      applyFlowParams();
    },
    onPreset: (kind) => {
      presetDirty = kind;
    },
    onAlphaChange: () => {
      presetDirty = 'airfoil';
    },
    onClearObstacles: () => {
      writeMask(device, buffers, new Uint8Array(NX * NY));
      maskEdited = true;
    },
    onResetFlow: () => {
      resetFlow();
    },
    onPauseToggle: () => {
      paused = !paused;
      return paused;
    },
    onSingleStep: () => {
      stepOnce = true;
    },
  });

  // --- pointer -> brush ---
  function toLattice(e: PointerEvent): { x: number; y: number } {
    const r = canvas.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * NX;
    const y = NY - ((e.clientY - r.top) / r.height) * NY;
    return { x: Math.max(0, Math.min(NX - 1, x)), y: Math.max(0, Math.min(NY - 1, y)) };
  }
  canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
  });
  canvas.addEventListener('pointerdown', (e) => {
    try {
      canvas.setPointerCapture(e.pointerId);
    } catch {
      // Synthetic or already-released pointers can't be captured; the
      // stroke still records normally.
    }
    stroke.erase = e.button === 2 || state.brushErase;
    stroke.points.push(toLattice(e));
  });
  canvas.addEventListener('pointermove', (e) => {
    if (e.buttons !== 0) {
      stroke.points.push(toLattice(e));
    }
  });
  canvas.addEventListener('pointerup', (e) => {
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
    lastStamp = null;
  });

  /** Stamp the queued stroke points (one small submit per stamp -- the
   *  brush uniform is rewritten between dispatches). */
  function flushStroke(): void {
    if (stroke.points.length === 0) return;
    const spacing = Math.max(1, state.brushRadius * 0.5);
    const stamps: { x: number; y: number }[] = [];
    for (const p of stroke.points) {
      if (lastStamp) {
        const dist = Math.hypot(p.x - lastStamp.x, p.y - lastStamp.y);
        const steps = Math.floor(dist / spacing);
        for (let s = 1; s <= steps; s++) {
          stamps.push({
            x: lastStamp.x + ((p.x - lastStamp.x) * s) / steps,
            y: lastStamp.y + ((p.y - lastStamp.y) * s) / steps,
          });
        }
        if (steps === 0) continue;
      } else {
        stamps.push(p);
      }
      lastStamp = stamps[stamps.length - 1] ?? p;
    }
    stroke.points.length = 0;
    for (const s of stamps) {
      writeBrushParams(device, buffers, {
        centerX: s.x,
        centerY: s.y,
        radius: state.brushRadius,
        paint: !stroke.erase,
        nx: NX,
        ny: NY,
      });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(brush.apply);
      pass.setBindGroup(0, brush.applyBindGroup);
      pass.dispatchWorkgroups(brush.workgroupsX, brush.workgroupsY);
      pass.end();
      device.queue.submit([encoder.finish()]);
      maskEdited = true;
    }
  }

  // --- frame loop ---
  let fps = 60;
  let lastTime = performance.now();
  let lastStatus = 0;

  function frame(now: number): void {
    fps = 0.95 * fps + 0.05 * (1000 / Math.max(1, now - lastTime));
    lastTime = now;

    if (presetDirty) {
      const preset = buildPreset(presetDirty);
      currentD = preset.d;
      lastPreset = presetDirty;
      writeMask(device, buffers, preset.mask);
      applyFlowParams(); // D changed -> tau changes
      maskEdited = true;
      presetDirty = null;
    }
    flushStroke();

    const encoder = device.createCommandEncoder();
    if (maskEdited) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(brush.diff);
      pass.setBindGroup(0, brush.diffBindGroup);
      pass.dispatchWorkgroups(brush.workgroupsX, brush.workgroupsY);
      pass.setPipeline(brush.reconcile);
      pass.setBindGroup(0, brush.reconcileBindGroups[stepCount % 2]);
      pass.dispatchWorkgroups(brush.workgroupsX, brush.workgroupsY);
      pass.end();
      maskEdited = false;
    }
    const substeps = paused ? (stepOnce ? 1 : 0) : K_SUBSTEPS;
    stepOnce = false;
    if (substeps > 0) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(lbm.pipeline);
      for (let k = 0; k < substeps; k++) {
        pass.setBindGroup(0, lbm.bindGroups[stepCount % 2]);
        pass.dispatchWorkgroups(lbm.workgroupsX, lbm.workgroupsY);
        stepCount++;
      }
      pass.end();
    }
    const rp = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    });
    rp.setPipeline(render.pipeline);
    rp.setBindGroup(0, render.bindGroup);
    rp.draw(3);
    rp.end();
    device.queue.submit([encoder.finish()]);

    if (now - lastStatus > 500) {
      lastStatus = now;
      const eff = tauSolution.clamped ? ` (eff ${tauSolution.reEffective.toFixed(0)})` : '';
      controls.setStatus(
        `fps    ${fps.toFixed(0)}\n` +
          `Re     ${state.re}${eff}\n` +
          `tau    ${tauSolution.tau.toFixed(4)}\n` +
          `D      ${currentD} cells (${lastPreset})\n` +
          `steps  ${stepCount}`,
      );
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  // Dev-only probe: read the moment fields back for verification tooling
  // (the Browser-pane screenshot path is unreliable here; numeric probes
  // are the ground truth).
  if (import.meta.env.DEV) {
    const readField = async (src: GPUBuffer): Promise<Float32Array> => {
      const staging = device.createBuffer({
        size: buffers.n * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(src, 0, staging, 0, buffers.n * 4);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(staging.getMappedRange().slice(0));
      staging.destroy();
      return data;
    };
    Object.assign(window, {
      __lbm: {
        nx: NX,
        ny: NY,
        steps: () => stepCount,
        readMoments: async () => ({
          rho: await readField(buffers.rho),
          ux: await readField(buffers.ux),
          uy: await readField(buffers.uy),
        }),
      },
    });
  }
}

main().catch((err: unknown) => {
  console.error('Fatal error during startup:', err);
  showFallback();
});
