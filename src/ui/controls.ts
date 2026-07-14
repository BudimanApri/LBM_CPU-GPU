// Thin DOM controls panel -- no framework, no solver knowledge. The panel
// mutates the shared ControlsState and pokes the handlers; main.ts owns all
// simulation consequences.
import type { ViewMode } from '../gpu/buffers.ts';
import { LATTICE_RESOLUTIONS, type LatticeResolution } from '../gpu/resolution.ts';

export type PresetKind = 'cylinder' | 'airfoil' | 'plate-normal' | 'plate-inclined' | 'step';

export interface ControlsState {
  re: number;
  u: number;
  periodicY: boolean;
  brushRadius: number;
  brushErase: boolean;
  cylinderDiameter: number;
  cylinderDiameterMax: number;
  nacaDigits: string;
  nacaAlphaDeg: number;
  viewMode: ViewMode;
  dyeEnabled: boolean;
  particlesEnabled: boolean;
  resolution: LatticeResolution;
  supportedResolutions: readonly LatticeResolution[];
}

export interface ControlsHandlers {
  onFlowParamsChange: () => void;
  onPreset: (kind: PresetKind) => void;
  onAlphaChange: () => void;
  onClearObstacles: () => void;
  onResetFlow: () => void;
  onPauseToggle: () => boolean;
  onSingleStep: () => void;
  onResolutionChange: (resolution: LatticeResolution) => void;
}

export interface ControlsApi {
  setStatus: (text: string) => void;
}

function row(parent: HTMLElement, label: string): HTMLDivElement {
  const div = document.createElement('div');
  div.className = 'ctl-row';
  if (label) {
    const span = document.createElement('span');
    span.className = 'ctl-label';
    span.textContent = label;
    div.appendChild(span);
  }
  parent.appendChild(div);
  return div;
}

function slider(
  parent: HTMLElement,
  label: string,
  min: number,
  max: number,
  step: number,
  value: number,
  onInput: (v: number) => void,
): HTMLInputElement {
  const r = row(parent, label);
  const input = document.createElement('input');
  input.type = 'range';
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(value);
  const readout = document.createElement('span');
  readout.className = 'ctl-value';
  readout.textContent = String(value);
  input.addEventListener('input', () => {
    const v = Number(input.value);
    readout.textContent = String(v);
    onInput(v);
  });
  r.appendChild(input);
  r.appendChild(readout);
  return input;
}

function button(parent: HTMLElement, label: string, onClick: () => void): HTMLButtonElement {
  const b = document.createElement('button');
  b.textContent = label;
  b.addEventListener('click', onClick);
  parent.appendChild(b);
  return b;
}

export function buildControls(
  root: HTMLElement,
  state: ControlsState,
  handlers: ControlsHandlers,
): ControlsApi {
  root.replaceChildren();

  const title = document.createElement('h1');
  title.textContent = 'LBM WIND TUNNEL';
  root.appendChild(title);

  // -- flow --
  const flow = row(root, '');
  flow.className = 'ctl-section';
  flow.textContent = 'FLOW';
  // Log slider for Re: 1..4 maps to 10^v.
  slider(root, 'Re (log)', 1, 4, 0.05, Math.log10(state.re), (v) => {
    state.re = Math.round(10 ** v);
    handlers.onFlowParamsChange();
  });
  slider(root, 'Inlet U', 0.02, 0.1, 0.005, state.u, (v) => {
    state.u = v;
    handlers.onFlowParamsChange();
  });
  const wallsRow = row(root, 'Walls');
  const walls = document.createElement('select');
  for (const [value, label] of [
    ['free-slip', 'free-slip'],
    ['periodic', 'periodic'],
  ]) {
    const opt = document.createElement('option');
    opt.value = value!;
    opt.textContent = label!;
    walls.appendChild(opt);
  }
  walls.value = state.periodicY ? 'periodic' : 'free-slip';
  walls.addEventListener('change', () => {
    state.periodicY = walls.value === 'periodic';
    handlers.onFlowParamsChange();
  });
  wallsRow.appendChild(walls);

  // -- obstacles --
  const obs = row(root, '');
  obs.className = 'ctl-section';
  obs.textContent = 'OBSTACLES';
  const presets = row(root, '');
  presets.className = 'ctl-buttons';
  button(presets, 'Cylinder', () => {
    handlers.onPreset('cylinder');
  });
  button(presets, 'Airfoil', () => {
    handlers.onPreset('airfoil');
  });
  button(presets, 'Plate', () => {
    handlers.onPreset('plate-normal');
  });
  button(presets, 'Plate ∠', () => {
    handlers.onPreset('plate-inclined');
  });
  button(presets, 'Step', () => {
    handlers.onPreset('step');
  });
  slider(root, 'Cyl. dia.', 8, state.cylinderDiameterMax, 2, state.cylinderDiameter, (v) => {
    state.cylinderDiameter = v;
    handlers.onPreset('cylinder');
  });
  const nacaRow = row(root, 'NACA');
  const naca = document.createElement('input');
  naca.type = 'text';
  naca.maxLength = 4;
  naca.value = state.nacaDigits;
  naca.addEventListener('change', () => {
    if (/^\d{4}$/.test(naca.value)) {
      state.nacaDigits = naca.value;
      handlers.onPreset('airfoil');
    } else {
      naca.value = state.nacaDigits;
    }
  });
  nacaRow.appendChild(naca);
  slider(root, 'AoA °', -15, 15, 0.5, state.nacaAlphaDeg, (v) => {
    state.nacaAlphaDeg = v;
    handlers.onAlphaChange();
  });

  // -- brush --
  const brush = row(root, '');
  brush.className = 'ctl-section';
  brush.textContent = 'BRUSH (drag on canvas; right-drag erases)';
  slider(root, 'Radius', 1, 30, 1, state.brushRadius, (v) => {
    state.brushRadius = v;
  });
  const eraseRow = row(root, 'Erase mode');
  const erase = document.createElement('input');
  erase.type = 'checkbox';
  erase.checked = state.brushErase;
  erase.addEventListener('change', () => {
    state.brushErase = erase.checked;
  });
  eraseRow.appendChild(erase);

  // -- view --
  const view = row(root, '');
  view.className = 'ctl-section';
  view.textContent = 'VIEW';
  const viewRow = row(root, 'Mode');
  const viewSelect = document.createElement('select');
  const viewOptions: readonly [ViewMode, string][] = [
    ['velocity', 'velocity'],
    ['vorticity', 'vorticity'],
    ['density', 'density'],
    ['dye', 'dye/smoke'],
  ];
  for (const [value, label] of viewOptions) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = label;
    viewSelect.appendChild(opt);
  }
  viewSelect.value = state.viewMode;
  viewSelect.addEventListener('change', () => {
    state.viewMode = viewSelect.value as ViewMode;
    // View mode and dye flags live in the params uniform -- the GPU only
    // sees them after a rewrite.
    handlers.onFlowParamsChange();
  });
  viewRow.appendChild(viewSelect);

  const dyeRow = row(root, 'Dye emitters');
  const dyeToggle = document.createElement('input');
  dyeToggle.type = 'checkbox';
  dyeToggle.checked = state.dyeEnabled;
  dyeToggle.addEventListener('change', () => {
    state.dyeEnabled = dyeToggle.checked;
    handlers.onFlowParamsChange();
  });
  dyeRow.appendChild(dyeToggle);

  const particlesRow = row(root, 'Particles');
  const particlesToggle = document.createElement('input');
  particlesToggle.type = 'checkbox';
  particlesToggle.checked = state.particlesEnabled;
  particlesToggle.addEventListener('change', () => {
    state.particlesEnabled = particlesToggle.checked;
  });
  particlesRow.appendChild(particlesToggle);

  // -- run control --
  const run = row(root, '');
  run.className = 'ctl-section';
  run.textContent = 'RUN';
  const resolutionRow = row(root, 'Resolution');
  const resolutionSelect = document.createElement('select');
  for (const value of LATTICE_RESOLUTIONS) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = value;
    opt.disabled = !state.supportedResolutions.includes(value);
    resolutionSelect.appendChild(opt);
  }
  resolutionSelect.value = state.resolution;
  resolutionSelect.addEventListener('change', () => {
    handlers.onResolutionChange(resolutionSelect.value as LatticeResolution);
  });
  resolutionRow.appendChild(resolutionSelect);
  const runRow = row(root, '');
  runRow.className = 'ctl-buttons';
  const pauseBtn = button(runRow, 'Pause', () => {
    pauseBtn.textContent = handlers.onPauseToggle() ? 'Resume' : 'Pause';
  });
  button(runRow, 'Step', () => {
    handlers.onSingleStep();
  });
  button(runRow, 'Reset flow', () => {
    handlers.onResetFlow();
  });
  button(runRow, 'Clear obst.', () => {
    handlers.onClearObstacles();
  });

  const status = document.createElement('pre');
  status.id = 'status';
  root.appendChild(status);

  return {
    setStatus: (text) => {
      status.textContent = text;
    },
  };
}
