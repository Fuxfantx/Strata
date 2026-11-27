import { useEffect, useState } from 'react';
import { Icon, type IconName } from './ui';
import { modelFamilyIcon } from '../lib/model-icons';

/**
 * A model family's mark: the brand's **own** file, else the house glyph.
 *
 * The official marks are binaries that cannot be authored here, so they live in `public/brand/` (see its README for
 * where each came from and the rule that they are never redrawn or recoloured).  The app asks for the file and falls
 * back to the sprite glyph when it is absent, so a checkout without them shows a neutral mark instead of a broken one.
 * A mark is scaled uniformly: the square logo fills the box, a wider lock-up keeps its proportions.
 */
const MARKS: Record<string, string> = {
  edge0: '/brand/edge0.png',        // Edge0's organisation avatar, 200 x 200
  hy4: '/brand/tencent-hy.png',     // the "Tencent Hy" lock-up for the Hy4 family, shown at its own proportions
};

export function BrandMark({ family, size = 20, className }: { family?: string; size?: number; className?: string }) {
  const mark = family ? MARKS[family.toLowerCase()] : undefined;
  const [found, setFound] = useState(false);
  useEffect(() => {
    setFound(false);
    if (!mark) return;
    let live = true;
    const probe = new Image();
    probe.onload = () => { if (live) setFound(true); };
    probe.src = mark;
    return () => { live = false; };
  }, [mark]);
  if (mark && found) {
    return (
      <img className={className} src={mark} alt="" height={size}
           style={{ height: size, width: 'auto', maxWidth: size * 4, objectFit: 'contain' }} />
    );
  }
  return <Icon name={modelFamilyIcon(family) as IconName} size={size} className={className} />;
}
