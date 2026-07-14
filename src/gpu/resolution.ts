import { supportsLbmResolution } from './buffers.ts';

export const LATTICE_RESOLUTIONS = ['512x256', '1024x512', '2048x1024'] as const;
export type LatticeResolution = (typeof LATTICE_RESOLUTIONS)[number];

export const DEFAULT_RESOLUTION: LatticeResolution = '1024x512';

const DIMENSIONS: Record<LatticeResolution, { nx: number; ny: number }> = {
  '512x256': { nx: 512, ny: 256 },
  '1024x512': { nx: 1024, ny: 512 },
  '2048x1024': { nx: 2048, ny: 1024 },
};

export function isLatticeResolution(value: string | null): value is LatticeResolution {
  return LATTICE_RESOLUTIONS.some((candidate) => candidate === value);
}

export function resolutionDimensions(value: LatticeResolution): { nx: number; ny: number } {
  return DIMENSIONS[value];
}

export function requestedResolution(search: string): LatticeResolution {
  const value = new URLSearchParams(search).get('resolution');
  return isLatticeResolution(value) ? value : DEFAULT_RESOLUTION;
}

export function supportedResolutions(device: GPUDevice): readonly LatticeResolution[] {
  return LATTICE_RESOLUTIONS.filter((value) => {
    const { nx, ny } = resolutionDimensions(value);
    return supportsLbmResolution(device, nx, ny);
  });
}
