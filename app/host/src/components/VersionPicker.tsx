import { useEffect, useRef, useState } from 'react';
import { tr } from '../lib/i18n';
import { Icon } from './ui';
import { FloatingMenu } from './FloatingMenu';
import { cx } from '../lib/format';
import type { CatalogVersion } from '../lib/native-catalog';
import { fitOn, recommendedVersions, versionFit, versionGroups, type CacheChoice, type FitHardware, type PcPick } from '../lib/native-fit';

/** The name a version goes by in the list: the quantization, with the maker only where the quantization alone would be ambiguous. */
export function versionName(v: CatalogVersion): string {
  if (v.custom && v.displayName) return v.displayName;      // the person's own build: the name they chose
  switch (v.sourceFamily) {
    case 'swift': return `Swift ${v.model}`;
    case 'unsloth': return `Unsloth ${v.model}`;
    case 'coder': return `Coder ${v.model}`;
    default: return v.model;
  }
}

/** Who made a version (Details only): the family name of the quantization, a proper name in every language. */
const MAKERS: Record<string, string> = { qwen: 'GSQ-RCO', coder: 'GSQ-RCO', swift: 'Swift', unsloth: 'Unsloth', gyro: 'Gyro' };
const makerOf = (v: CatalogVersion): string | undefined => v.custom ? 'Local files' : MAKERS[v.sourceFamily];

const gb = (n: number) => `${Number.isInteger(n) ? n : n.toFixed(1)} GB`;
const whole = (n: number) => `${Math.round(n)} GB`;

/**
 * Where the versions of the chosen base model are listed, each exactly once. At the top: the versions we recommend (the pick for this PC first; no "Recommended" tag since 11-02, the order says it) and the versions already downloaded (tagged "Downloaded"). Below, folded under "More
 * versions": all the others. A row is the name, small tags, a word only where one is needed ("may run slower" for a version that does
 * not fit in memory; a version that fits says nothing: see `versionFit`) and the size. The maker and the
 * memory it needs are behind "Show details" (open from the start in Expert mode). The other base models are not in this list: they
 * appear when picked at the top of the page.
 */
export function VersionPicker({ versions, selectedId, hardware, onSelect, onOwnGguf, downloaded, showRecommended = true, moreLabel, pick, cache, details: detailsOpen = false }: {
  /** The versions of the chosen base model. */
  versions: CatalogVersion[];
  selectedId: string;
  hardware: FitHardware | null | undefined;
  onSelect: (id: string) => void;
  onOwnGguf: () => void;
  /** The versions already on this computer (ready to load): each gets a small "Downloaded" tag. */
  downloaded?: ReadonlySet<string>;
  /** False where the recommended ones are shown elsewhere (the first-run page): the menu then holds only the others. */
  showRecommended?: boolean;
  /** What the button says while the selected version is one of the recommended ones shown elsewhere. */
  moreLabel?: string;
  /** The engine's own pick for this PC (catalog.recommendation). */
  pick?: PcPick | null;
  /** The conversation context and cache format the person chose: the fit words follow them, as the This computer card does. */
  cache?: CacheChoice | null;
  /** Start with the technical detail shown (Expert mode). */
  details?: boolean;
}) {
  const trigger = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const [details, setDetails] = useState(detailsOpen);
  useEffect(() => setDetails(detailsOpen), [detailsOpen]);
  const selected = versions.find(v => v.id === selectedId);
  // Where the recommended ones are shown outside the menu, the menu holds only the others.
  const rows = recommendedVersions(versions, pick);
  const apart = new Set(showRecommended ? [] : rows.map(x => x.v.id));
  const recommendedIds = rows.map(x => x.v.id);
  const { top, more } = versionGroups(versions, apart, downloaded, recommendedIds);
  const shown = selected && !apart.has(selected.id) ? selected : undefined;
  // The folded part opens by itself when it holds the chosen version or when there is nothing above it.
  const [moreOpen, setMoreOpen] = useState(false);
  const foldedOpen = moreOpen || !top.length || more.some(v => v.id === selectedId);

  const need = (v: CatalogVersion, technical: boolean) => {
    const size = typeof v.fit_gb === 'number' && Number.isFinite(v.fit_gb) ? v.fit_gb : fitOn(v, hardware, true, cache).memoryGb;       // the files without the lookup table: the figure the label uses
    if (technical) return tr("{0} RAM + VRAM", [gb(size)]);
    return tr("Needs {0} of memory", [whole(size)]);
  };
  const row = (v: CatalogVersion) => {
    const on = v.id === selectedId, have = !!downloaded?.has(v.id) && !v.custom;       // a custom build has its own "Custom" tag instead of "Downloaded": its files are the person's own
    const state = versionFit(v, hardware, cache);
    const technical = [makerOf(v), need(v, true), v.credit ? tr(v.credit) : ''].filter(Boolean).join(' · ');
    return <button key={v.id} role="menuitemradio" aria-checked={on} className={cx('version-row', on && 'is-on')} onClick={() => { onSelect(v.id); setOpen(false); }}>
      <span className="version-row__check">{tr(on ? <Icon name="check" size={14}/> : null)}</span>
      <span className="version-row__body">
        <span className="version-row__title"><strong>{versionName(v)}</strong>{tr(have || v.offered === false || v.custom ? <span className="version-row__tags">{tr(v.custom ? <span className="version-tag version-tag--custom">{tr("Custom")}</span> : null)}{tr(have ? <span className="version-tag version-tag--have">{tr("Downloaded")}</span> : null)}{tr(v.offered === false ? <span className="version-tag">{tr("No longer offered")}</span> : null)}{tr(have && v.update_available ? <span className="version-tag">{tr("Update available")}</span> : null)}</span> : null)}</span>
        {tr(state === 'ssd' ? <span className="version-row__fit" data-fit={state}>{tr('may run slower')}</span> : null)}
        {tr(details ? <span className="version-row__tech">{technical}</span> : null)}
      </span>
      {/* an imported pack has nothing of ours to download, so it has no download size to state */}
      {!v.import_only ? <span className="version-row__size">{have || v.custom ? gb(v.download_gb) : tr("{0} download", [gb(v.download_gb)])}</span> : null}
    </button>;
  };
  // The open menu opens on the chosen version, not at the top of a long list.
  const menu = useRef<HTMLDivElement>(null);
  useEffect(() => { if (open) menu.current?.querySelector('.is-on')?.scrollIntoView({ block: 'nearest' }); }, [open]);
  return <>
    <button ref={trigger} type="button" className="version-trigger" aria-haspopup="menu" aria-expanded={open} aria-label={tr("Model version")} onClick={() => setOpen(v => !v)}>
      <strong>{shown ? versionName(shown) : tr(moreLabel ?? "Choose a version")}</strong>{tr(shown?.custom ? <span className="version-tag version-tag--custom">{tr("Custom")}</span> : shown && downloaded?.has(shown.id) ? <span className="version-tag version-tag--have">{tr("Downloaded")}</span> : null)}
      <span className="version-trigger__need muted">{tr(shown ? need(shown, details) : '')}</span>
      <Icon name="chevron" size={14} className="version-trigger__chev"/>
    </button>
    {tr(open ? <FloatingMenu anchor={trigger.current} title={tr("Model version")} width={520} onClose={() => setOpen(false)}>
      <div className="version-menu" ref={menu}>
        {tr(top.length ? <div className="version-section" role="group" aria-label={tr("Model version")}>{tr(top.map(row))}</div> : null)}
        {tr(more.length ? <div className="version-section" role="group" aria-label={tr("More versions")}>
          {tr(top.length ? <button role="menuitem" className="version-more" aria-expanded={foldedOpen} onClick={() => setMoreOpen(!foldedOpen)}><span>{tr("More versions")} <span className="muted">{more.length}</span></span><Icon name="chevron" size={14} className={cx('version-more__chev', foldedOpen && 'is-open')}/></button> : null)}
          {tr(foldedOpen ? <div className="version-more__rows">{tr(more.map(row))}</div> : null)}
        </div> : null)}
        <div className="version-foot">
          <button role="menuitem" className="version-foot__btn" onClick={() => { setOpen(false); onOwnGguf(); }}><Icon name="folder" size={14}/>{tr("Use local model files (.strata or .gguf)…")}</button>
          <button role="menuitem" className="version-foot__btn" aria-pressed={details} onClick={() => setDetails(d => !d)}>{tr(details ? "Hide details" : "Show details")}</button>
        </div>
      </div>
    </FloatingMenu> : null)}
  </>;
}
