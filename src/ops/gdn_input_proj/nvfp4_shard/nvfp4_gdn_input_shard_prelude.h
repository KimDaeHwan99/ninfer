#pragma once

// Included first by every translation unit of this directory: the family's headers, processed
// once under their own names before nvfp4_gdn_input_shard_names.h renames anything.

#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_a4_tma_launch.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4_shard/nvfp4_gdn_input_shard.h"
#include "ops/linear/nvfp4/nvfp4_layout.h"

namespace ninfer::ops::detail {

// The shard's qkv|z section output, in the parent's section order.
using Nvfp4GdnInputShardOutput = LinearBf16SegmentedOutput<5120, 3072>;

static_assert(kNvfp4GdnInputShardRows == 5120 + 3072);
// Every row tile of the parent's routes (at most 128 rows) divides each shard section.
static_assert((5120 % 128) == 0 && (3072 % 128) == 0);

} // namespace ninfer::ops::detail
