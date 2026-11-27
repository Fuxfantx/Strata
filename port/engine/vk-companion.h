// vk-companion.h - write a companion GGUF that replaces a few tensors of a base pack.
//
// The loader composes such a file over the base pack when it is handed over as an extra file (`--compat`, or the
// engine's own automatic pass): a tensor in the companion replaces the base pack's tensor of the same name, and the
// companion is marked `strata.companion = true` so nothing mistakes it for a model.
//
// One user of this is measured: `requant-tensor` builds a lossy companion for an A/B.  The other is the engine's
// BF16 -> F16 pass, which is not lossy for weights (a BF16 value with a 7-bit mantissa is exactly representable in
// F16 whenever its magnitude is in F16's normal range) and is worth 19% of the prefill on the Q2 pack
// (port/docs/PERFORMANCE.md 5.5.5).
#pragma once

#include "ggml.h"

#include <string>
#include <vector>

namespace strata {

struct CompanionPick {
    std::string     name;   // tensor name in the base pack
    enum ggml_type  type;   // destination type
};

// Every tensor name in `base`, in file order.
std::vector<std::string> companion_tensor_names(const std::string & base);

// Every tensor of type `from` in `base`, to be written as `to`.  Empty when the pack holds none.
std::vector<CompanionPick> companion_picks_of_type(const std::string & base, enum ggml_type from, enum ggml_type to);

// A row slice of one tensor: rows [0, keep_rows) keep `name`, the rest are written as `low_name` with the same type.
// This is the storage half of a row-split matmul - the two halves then live in two buffers and can be computed by two
// backends.  A slice costs no re-quantisation: rows of a row-major quantised tensor are whole blocks, so the bytes
// move as they are.
struct CompanionSlice {
    std::string name;
    std::string low_name;
    // Which dimension the slice runs along: 1 keeps the first `keep_rows` rows and puts the rest in `low_name` (the
    // row split of a matmul), 2 keeps the first `keep_rows` experts of a 3-D expert stack (an expert split).
    int         dim = 1;
    int64_t     keep_rows;
};

// Write `picks` out of `base` into `out`, converting each to its destination type.  A tensor whose ne0 is not a
// multiple of the destination block size is skipped with a message rather than written wrong.  `slices` are applied
// after the picks and keep their source type.  Returns false and fills `err` on failure.
bool write_companion(const std::string & base, const std::string & out,
                     const std::vector<CompanionPick> & picks, std::string & err,
                     const std::vector<CompanionSlice> & slices = {});

// Does `companion` record that it was built from `base`?  The engine names a companion after the pack, so a pack
// replaced under the same name would otherwise be composed with tensors converted from the old one.  A companion that
// carries no such record answers true: the check is for the ones this code wrote.
bool companion_matches(const std::string & companion, const std::string & base);

// Every 2-D tensor in `path` as (name, ne0, ne1), for choosing what to row split.
struct CompanionTensorInfo { std::string name; int64_t ne0 = 0; int64_t ne1 = 0; int64_t ne2 = 1; int64_t ne3 = 1; };
std::vector<CompanionTensorInfo> companion_tensor_list(const std::string & path);

// ne0/ne1 of the named tensor in `path`, which may be a pack or a companion.  False when it is absent.
bool companion_tensor_rows(const std::string & path, const std::string & name, int64_t & ne0, int64_t & ne1);

} // namespace strata
