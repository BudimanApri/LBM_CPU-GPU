import './style.css';
import { initGPU, observeCanvasResize } from './gpu/context.ts';
import {
  PARTICLE_COUNT,
  createLbmBuffers,
  seedParticlesRandom,
  writeBrushParams,
  writeMask,
  writeParams,
  writeUniformEquilibrium,
  type LbmBuffers,
} from './gpu/buffers.ts';
import {
  createBrushPipelines,
  createDyePipeline,
  createForcePipelines,
  createLbmPipeline,
  createParticlePipelines,
  createRenderPipeline,
} from './gpu/pipelines.ts';
import { clampU, solveTauForRe, type TauSolution } from './solver/units.ts';
import {
  backwardStepPreset,
  cylinderPreset,
  maskFrontalHeight,
  nacaPreset,
  plateInclinedPreset,
  plateNormalPreset,
  type PresetResult,
} from './geometry/presets.ts';
import { buildControls, type ControlsState, type PresetKind } from './ui/controls.ts';
import { formatHud } from './ui/hud.ts';
import { createStripChart } from './ui/chart.ts';

// Lattice resolution: 1024x512, matching the Phase 4 gate configuration
// (60fps at K=2) directly rather than testing it separately. The
// 512x256/2048x1024 selector arrives with the Phase 6 performance work.
// Keep the CSS aspect-ratio of #gpu-canvas in sync (2 / 1).
const NX = 1024;
const NY = 512;
const K_MIN = 1;
const K_MAX = 8;
const K_DEFAULT = 3;
const TARGET_FRAME_MS = 1000 / 60;

const CYLINDER_DIAMETER_DEFAULT = 48;
const PLATE_HEIGHT = 96;
const PLATE_LENGTH = 128;
const PLATE_ANGLE_DEG = 30;
const STEP_HEIGHT = Math.round(NY / 3);
const BRUSH_RADIUS_DEFAULT = 12;
// Force instrumentation: reduce + async readback every N frames (CLAUDE.md:
// "async readback every 10 frames", never awaited in the frame loop).
const FORCE_READBACK_INTERVAL = 10;
// Debounce for the mask -> D (frontal height) readback during a brush drag;
// finalized immediately on pointerup. Presets/clear supply D directly.
const MASK_D_DEBOUNCE_MS = 100;

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
  const dye = createDyePipeline(device, buffers);
  const particles = createParticlePipelines(device, buffers, format);
  const forces = createForcePipelines(device, buffers);
  const render = createRenderPipeline(device, buffers, format);
  seedParticlesRandom(device, buffers);

  // Persistent staging buffer for the async force readback (8 bytes: Fx, Fy).
  const forceStaging = device.createBuffer({
    label: 'force-staging',
    size: 8,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  // Cd/Cl strip chart with the live Strouhal readout (the signature
  // instrument). Appended below the controls panel.
  const chart = createStripChart();

  // --- simulation state ---
  // Vorticity is the default view: CLAUDE.md calls it out as "the money
  // shot for vortex streets," and the Definition of Done wants a visible
  // vortex street within the first 10 seconds of loading.
  const state: ControlsState = {
    re: 100,
    u: 0.05,
    periodicY: false,
    brushRadius: BRUSH_RADIUS_DEFAULT,
    brushErase: false,
    cylinderDiameter: CYLINDER_DIAMETER_DEFAULT,
    nacaDigits: '4412',
    nacaAlphaDeg: 0,
    viewMode: 'vorticity',
    dyeEnabled: false,
    particlesEnabled: true,
  };
  let stepCount = 0; // parity: f[stepCount % 2] holds the current state
  let dyeStepCount = 0; // parity: dye[dyeStepCount % 2] holds the current state
  let paused = false;
  let stepOnce = false;
  let currentK = K_DEFAULT; // solver substeps per frame, adapted in frame()
  let currentD = state.cylinderDiameter;
  let obstacleLabel = 'cylinder'; // HUD tag; 'brush' once hand-painted
  let tauSolution: TauSolution = solveTauForRe(state.re, state.u, currentD);
  // Force instrumentation state.
  let frameCounter = 0;
  let forceReadbackInFlight = false;
  let latestCd = 0;
  let latestCl = 0;
  let latestSt: number | null = null;
  // Debounced mask -> D readback (brush edits only).
  let maskDTimer: number | null = null;
  let maskDReadbackInFlight = false;
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
      dyeEnabled: state.dyeEnabled,
      viewMode: state.viewMode,
      stepIndex: stepCount,
      substeps: currentK,
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
    // The 5% transverse seed starts the Karman instability within a couple
    // of shedding periods instead of waiting for roundoff to break the
    // symmetric start (see writeUniformEquilibrium).
    writeUniformEquilibrium(device, buffers, (stepCount % 2) as 0 | 1, 1, state.u, 0, 0.05);
    // Drop force history so the mean/Strouhal reflect only the new transient.
    chart.clear();
    latestSt = null;
  }

  // Brush edits have no analytic D, so the frontal height is read back from
  // the mask. This is an event-triggered readback (not part of the steady
  // 60fps loop, so it respects "no readbacks in the frame loop"), debounced
  // during a drag and finalized on pointerup. D feeds both the Re->tau solve
  // and the Cd/Cl normalization, so applyFlowParams runs when it changes.
  function readbackMaskD(): void {
    if (maskDReadbackInFlight) {
      maskDTimer = window.setTimeout(readbackMaskD, MASK_D_DEBOUNCE_MS);
      return;
    }
    maskDReadbackInFlight = true;
    const staging = device.createBuffer({
      size: buffers.n * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffers.mask, 0, staging, 0, buffers.n * 4);
    device.queue.submit([encoder.finish()]);
    void staging.mapAsync(GPUMapMode.READ).then(
      () => {
        const mask = new Uint32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        staging.destroy();
        const d = maskFrontalHeight(mask, NX, NY);
        obstacleLabel = 'brush';
        if (d > 0 && d !== currentD) {
          currentD = d;
          applyFlowParams();
        }
        maskDReadbackInFlight = false;
      },
      () => {
        maskDReadbackInFlight = false;
      },
    );
  }

  function scheduleMaskD(): void {
    if (maskDTimer !== null) clearTimeout(maskDTimer);
    maskDTimer = window.setTimeout(readbackMaskD, MASK_D_DEBOUNCE_MS);
  }

  function finalizeMaskD(): void {
    if (maskDTimer !== null) {
      clearTimeout(maskDTimer);
      maskDTimer = null;
    }
    readbackMaskD();
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
      obstacleLabel = 'none';
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

  // Instrument strip: Cd/Cl chart with the live Strouhal readout.
  const instr = document.createElement('div');
  instr.className = 'ctl-section';
  instr.textContent = 'FORCES';
  panel.appendChild(instr);
  panel.appendChild(chart.canvas);

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
    // Finalize D from the completed stroke (CLAUDE.md: always finalized on
    // pointerup). Guarded so a click with no stamps still resolves cheaply.
    finalizeMaskD();
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
    // A stamp changed the mask; debounce a D readback during the drag.
    scheduleMaskD();
  }

  // --- frame loop ---
  let fps = 60;
  let mlups = 0;
  let lastTime = performance.now();
  let lastStatus = 0;
  // Basic adaptive K (CLAUDE.md: rolling average, adjust to hold 60fps;
  // the full hysteresis-tuned version is Phase 6). Frame time under vsync
  // sits pinned at the display period even with the GPU nearly idle, so
  // "comfortably below target" can never fire on a 60Hz screen -- instead,
  // grow whenever the frame rate is holding the 60fps budget (small
  // tolerance for timer jitter) and shrink once frames actually start
  // missing it. The 30-frame cadence plus the EMA damp the oscillation at
  // the boundary. (currentK itself is declared with the sim state above --
  // applyFlowParams packs it into the params uniform.)
  let frameTimeAvg = TARGET_FRAME_MS;
  let framesSinceKAdjust = 0;

  function frame(now: number): void {
    const dt = Math.max(1, now - lastTime);
    lastTime = now;
    fps = 0.95 * fps + 0.05 * (1000 / dt);
    frameTimeAvg = 0.9 * frameTimeAvg + 0.1 * dt;
    framesSinceKAdjust++;
    if (framesSinceKAdjust > 30) {
      framesSinceKAdjust = 0;
      if (frameTimeAvg > TARGET_FRAME_MS * 1.25 && currentK > K_MIN) {
        currentK--;
        applyFlowParams(); // substeps lives in the params uniform
      } else if (frameTimeAvg < TARGET_FRAME_MS * 1.05 && currentK < K_MAX) {
        currentK++;
        applyFlowParams();
      }
    }

    if (presetDirty) {
      const preset = buildPreset(presetDirty);
      currentD = preset.d;
      obstacleLabel = presetDirty;
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
    const substeps = paused ? (stepOnce ? 1 : 0) : currentK;
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
      mlups = 0.9 * mlups + 0.1 * ((buffers.n * substeps) / (dt / 1000) / 1e6);
    }

    // -- momentum-exchange forces (every N frames, async readback) --
    // Recorded after the substeps so it reads the freshest post-collision
    // buffer f[stepCount % 2]. Two separate compute passes: WebGPU
    // synchronizes storage writes across the pass boundary, so the reduce
    // pass sees the accumulate pass's partials. The copy+map is only recorded
    // when no readback is already in flight (mapping a mapped buffer, or
    // copying into it, would be a validation error).
    let readbackThisFrame = false;
    const measuredStep = stepCount;
    if (frameCounter % FORCE_READBACK_INTERVAL === 0 && !forceReadbackInFlight) {
      const acc = encoder.beginComputePass();
      acc.setPipeline(forces.accumulate);
      acc.setBindGroup(0, forces.accumulateBindGroups[stepCount % 2]);
      acc.dispatchWorkgroups(forces.workgroupsX, forces.workgroupsY);
      acc.end();
      const red = encoder.beginComputePass();
      red.setPipeline(forces.reduce);
      red.setBindGroup(0, forces.reduceBindGroup);
      red.dispatchWorkgroups(1);
      red.end();
      encoder.copyBufferToBuffer(buffers.forceResult, 0, forceStaging, 0, 8);
      readbackThisFrame = true;
    }
    frameCounter++;

    // Dye advects once per rendered frame (a visual smoke effect, not part
    // of the physics substepping) using the freshest post-step velocity.
    if (state.dyeEnabled) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(dye.pipeline);
      pass.setBindGroup(0, dye.bindGroups[dyeStepCount % 2]);
      pass.dispatchWorkgroups(dye.workgroupsX, dye.workgroupsY);
      pass.end();
      dyeStepCount++;
    }
    if (state.particlesEnabled) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(particles.computePipeline);
      pass.setBindGroup(0, particles.computeBindGroup);
      pass.dispatchWorkgroups(particles.workgroups);
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
    rp.setBindGroup(0, render.bindGroups[dyeStepCount % 2]);
    rp.draw(3);
    if (state.particlesEnabled) {
      rp.setPipeline(particles.renderPipeline);
      rp.setBindGroup(0, particles.renderBindGroup);
      rp.draw(6, PARTICLE_COUNT);
    }
    rp.end();
    device.queue.submit([encoder.finish()]);

    // Kick off the force readback without awaiting -- the frame loop never
    // blocks on the GPU (CLAUDE.md's readback discipline). Cd/Cl/St are
    // updated in the .then() on a later tick; the in-flight flag prevents a
    // second copy into a buffer that is still mapped.
    if (readbackThisFrame) {
      forceReadbackInFlight = true;
      void forceStaging.mapAsync(GPUMapMode.READ).then(
        () => {
          const f = new Float32Array(forceStaging.getMappedRange().slice(0));
          forceStaging.unmap();
          // Cd = 2 Fx / (rho0 U^2 D), Cl = 2 Fy / (...); rho0 = 1.
          const denom = state.u * state.u * currentD;
          if (denom > 0) {
            latestCd = (2 * f[0]!) / denom;
            latestCl = (2 * f[1]!) / denom;
            chart.push({ step: measuredStep, cd: latestCd, cl: latestCl });
            latestSt = chart.strouhal(currentD, state.u);
          }
          forceReadbackInFlight = false;
        },
        () => {
          forceReadbackInFlight = false;
        },
      );
    }

    if (now - lastStatus > 500) {
      lastStatus = now;
      chart.render();
      // Report the time-mean coefficients: Cd's target is a mean, and the
      // average cancels the domain acoustic ripple on the instantaneous
      // force. Fall back to the latest sample before the buffer fills.
      const avg = chart.mean();
      controls.setStatus(
        formatHud({
          fps,
          mlups,
          kSubsteps: currentK,
          re: state.re,
          reEffective: tauSolution.reEffective,
          tau: tauSolution.tau,
          d: currentD,
          presetLabel: obstacleLabel,
          steps: stepCount,
          cd: avg ? avg.cd : latestCd,
          cl: avg ? avg.cl : latestCl,
          st: latestSt,
        }),
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
        // Coefficients for scripted validation (VALIDATION.md reads these
        // from the running app): instantaneous, plus the time-mean and sample
        // count that back the HUD's Cd/Cl readout.
        coefficients: () => ({
          cd: latestCd,
          cl: latestCl,
          st: latestSt,
          d: currentD,
          mean: chart.mean(),
          samples: chart.sampleCount(),
        }),
      },
    });
  }
}

main().catch((err: unknown) => {
  console.error('Fatal error during startup:', err);
  showFallback();
});
