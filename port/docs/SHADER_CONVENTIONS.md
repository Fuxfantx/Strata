# Shader conventions

Every compute shader that the backend can dispatch follows four rules. They exist because the backend has to know
three things about a shader before it can record a dispatch cheaply and correctly: which bindings it reads and
writes, where its scalars come from, and how much of its control flow depends on runtime values.

These rules apply to **every shader that is reachable from a graph**, not only to newly written ones. A shader that
comes in with an upstream merge is subject to the same check as a local one: either it follows the rules, or the
exception is written down in the shader with a reason (the `magic-div-not-applicable` comment is the existing
example).

## 1. Declare the shader in `ggml-vulkan-shader-meta.h`

Add one entry:

```cpp
{ "my_shader_prefix", VK_SHADER_MASK_DERIVE, true },
```

- the mask states the **roles**: bit `i` set means binding `i` is written by the shader, and the bits left clear
  are the bindings it only reads. Use an explicit mask when the shader writes a binding that the SPIR-V decoration
  does not mark; otherwise `VK_SHADER_MASK_DERIVE` reads the `NonWritable` decoration, which is the same evidence
  taken from the compiled module. Either way the backend must be able to tell reads from writes without parsing the
  GLSL: it plans descriptor reuse, barriers and buffer lifetimes from that mask.
- `param_block` states that the shader reads its scalars from the per-dispatch parameter block (rule 2).

The table is consulted when the pipeline is created, before the older name-prefix lists, so a declared shader can
never be caught by a prefix that was meant for another family.

## 2. Take scalars from the parameter block

```glsl
#define PARAM_BLOCK_BINDING 12
layout (binding = PARAM_BLOCK_BINDING, std430) readonly buffer parameter
{
    uint ncols;
    uint stride;
} p;
```

No `layout (push_constant)`. The host writes the struct into slot `n` of the parameter block, binds the block with a
dynamic offset, and records no push constants, so the recorded command stream does not depend on the token. The C++
struct and the GLSL block must list the same fields in the same order; check them field by field before landing.

## 3. Bake what is fixed, avoid runtime division

- A value that never changes for a pipeline variant belongs in a `#define` or a specialization constant, not in the
  block. The block is for values that change per dispatch.
- Integer division and modulo by a runtime divisor cost many cycles. The host computes a magic multiplier and shift
  once (at compile time when the divisor is fixed), and the shader uses `magic_div.glsl`:

```glsl
#include "magic_div.glsl"

const uint q = u32_div_magic(idx, mp, shift);   // idx / d
const uint r = u32_mod_magic(idx, d, mp, shift); // idx % d
```

`mul_mat_vec_base.glsl` already carries the same magic values (`ne12mp`, `ne12L`, `bc2mp`, ...) for its three
broadcast divides; new shaders should follow that shape instead of dividing directly.

### When the magic form does not apply

The pair from `init_pushconst_fastdiv` is the wrapping 32 bit form: `x / d == (x * mp) >> shift` holds only while
`x` stays below `1 << shift`. Above that bound the product overflows and the result is wrong, and the error is small
enough to pass an oracle test that never reaches the bound.

Before using the macro, state the largest value the index can take and compare it with `1 << shift`:

- bounded indices (a row inside a fixed tile, a head inside `ne02`, a broadcast index) are fine;
- indices that scale with the row count, the token position or the batch (`rowx`, a flat element index) are not,
  because at a long context or a large batch they pass the bound.

A shader whose index cannot be bounded keeps the plain divide and says so in a comment. Turning the macro into a form
that holds for every `x < 2^32` needs the 64 bit product (`(uint64_t(x) * mp) >> (32 + shift)` with a matching `mp`),
which is a separate change; until then a divide that is written plainly is correct and a magic divide that overflows
is not.

### Checking the whole tree

`port/tools/audit-shaders.py` lists every live shader that divides by a parameter block value, so a new one cannot
slip in unnoticed. It also lists the live sources that are not declared in `ggml-vulkan-shader-meta.h`, because an
undeclared shader is one whose roles the backend has to guess from a name prefix.

## 4. Keep the control flow uniform where the shape allows it

A runtime `if` whose condition is the same for a whole workgroup is cheap; one whose condition varies inside a
subgroup costs divergence, and on some drivers it also blocks the compiler from keeping loads in flight. The rules
of thumb, in order of preference:

- a condition that is fixed for a pipeline variant belongs in a specialization constant, and the code inside stays
  unrolled (this is what `#define` and the `ALIGNED`-style constants already do);
- a condition that depends on the dispatch shape (a partial tile at the edge of the matrix) stays a runtime `if`,
  but should test the tile bounds once per workgroup rather than once per element;
- an early `return` for workgroups that have nothing to do is correct and keeps the grid uniform, but the condition
  must be derivable from the block, otherwise the branch is unpredictable.

The same reasoning applies to loops: a loop whose trip count is a pipeline constant is unrolled by the compiler, a
loop whose trip count comes from the block is not, so a value that is constant for a variant does not belong in the
block (rule 3, first bullet).

