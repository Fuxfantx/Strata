import type { IconName } from '../components/ui';

/** The glyph a model family is drawn with: its own where the brand has one, the model library otherwise. */
export function modelFamilyIcon(family?: string): IconName {
  const name = (family ?? '').toLowerCase();
  if (/^qwen/.test(name)) return 'family-qwen';
  if (name === 'edge0') return 'family-edge0';
  if (name === 'hy4' || name === 'hy') return 'family-hy';
  return 'models';
}
