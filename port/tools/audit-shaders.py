"""Audit every Vulkan shader against the three rules in port/docs/SHADER_CONVENTIONS.md.

Rule 1: declared in ggml-vulkan-shader-meta.h (prefix match).
Rule 2: scalars from the parameter block, no push_constant block.
Rule 3: no integer division or modulo by a runtime divisor (a value that comes from the parameter block).
"""
import re, pathlib, collections, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SH = ROOT / "port/llama-overlay-vanilla/ggml/src/ggml-vulkan/vulkan-shaders"
GEN = SH / "vulkan-shaders-gen.cpp"
META = ROOT / "port/llama-overlay-vanilla/ggml/src/ggml-vulkan/ggml-vulkan-shader-meta.h"

gen = GEN.read_text(encoding="utf-8", errors="replace")
names = re.findall(r'string_to_spv\("([A-Za-z0-9_]+)",\s*"([A-Za-z0-9_.]+)"', gen)
# strip the type/variant suffixes the generator appends, giving the family prefix
def prefix(n):
    return re.sub(r'_(f16|f32|bf16|q8_0|q4_0|i64|i32|f16acc|fp32|cm1|cm2)$', '', n)

fam = collections.defaultdict(set)     # prefix -> sources
for n, src in names:
    fam[prefix(n)].add(src)

meta = META.read_text(encoding="utf-8", errors="replace")
declared = re.findall(r'\{\s*"([^"]+)"\s*,', meta)

def is_declared(p):
    return any(p.startswith(d) for d in declared if d) or any(p == d for d in declared if d)

# rule 2 and 3 per source
push, block, divs = set(), set(), collections.defaultdict(list)
for f in sorted(SH.glob("*.comp")) + sorted(SH.glob("*.glsl")):
    t = f.read_text(encoding="utf-8", errors="replace")
    if re.search(r'layout\s*\(push_constant\)', t):
        push.add(f.name)
    if re.search(r'layout\s*\(binding\s*=\s*PARAM_BLOCK_BINDING', t):
        block.add(f.name)
    for m in re.finditer(r'[/%]\s*p\.[A-Za-z_][A-Za-z0-9_]*', t):
        divs[f.name].append(m.group(0))

undeclared = sorted(p for p in fam if not is_declared(p))
blocked_on_push = sorted(s for s in {s for p in fam for s in fam[p]} if s in push and s not in block)

# A source is live when the generator registers at least one pipeline from it.  Dead sources are not built, so the
# conventions do not apply to them and they are left out of the denominators.
# Some sources are selected through a variable (mul_mm.comp and mul_mmq.comp go through source_name), so a plain
# literal anywhere in the generator also makes a source live.
literal = set(re.findall(r'"([A-Za-z0-9_]+\.comp)"', gen))
live = {src for srcs in fam.values() for src in srcs} | literal
dead = sorted({f.name for f in SH.glob("*.comp")} - live)
print(f"families: {len(fam)}  declared: {len(fam) - len(undeclared)}  undeclared: {len(undeclared)}")
print(f"live shader sources: {len(live)}  dead sources (not registered): {len(dead)}")
print(f"live sources with a push constant block: {len(push & live)}")
print(f"live sources using the parameter block: {len(block & live)}")
print(f"live sources dividing by a block value (rule 3): {len(set(divs) & live)}")
print(f"dead sources: {', '.join(dead[:10])}{' ...' if len(dead) > 10 else ''}")

# The generator's opt-in set is the authoritative "migrated" list, because most sources take the block through a head.
m = re.search(r"use_param_block = \{(.*?)\};", gen, re.S)
wl = set(re.findall(r'"([A-Za-z0-9_.]+)"', m.group(1))) if m else set()
print(f"whitelisted sources (rule 2 done): {len(wl & live)} of {len(live)} live")
print(f"rule 3 live sources: {sorted(set(divs) & live)}")
# Every live divide site, so a new one cannot slip in unnoticed.  The bound column is what decides whether the
# magic form may be used at all (see SHADER_CONVENTIONS.md).
MARK = "magic-div-not-applicable"
unmarked = {}
for src in sorted(set(divs) & live):
    text = (SH / src).read_text(encoding="utf-8", errors="replace").splitlines()
    bad = []
    for i, ln in enumerate(text):
        if re.search(r"[/%]\s*p\.[A-Za-z_]", ln):
            window = "\n".join(text[max(0, i - 12):i])
            if MARK not in window and MARK not in ln:
                bad.append(ln.strip())
    if bad:
        unmarked[src] = bad
print("rule 3 sites:")
for src in sorted(set(divs) & live):
    print(f"  {src}: {sorted(set(divs[src]))}")
print(f"rule 3 undocumented divide sites: {sum(len(v) for v in unmarked.values())}")
for src, bad in unmarked.items():
    print(f"  {src}: {bad[:3]}")
print("\nundeclared families (rule 1):")
for p in undeclared[:40]:
    print(f"  {p:<28} {sorted(fam[p])}")
if len(undeclared) > 40:
    print(f"  ... and {len(undeclared) - 40} more")
print("\nsources dividing by a block value (rule 3):")
for s, ds in list(divs.items())[:12]:
    print(f"  {s:<28} {sorted(set(ds))}")
