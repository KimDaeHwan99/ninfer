#pragma once

// Included first by every translation unit of this directory: the family's headers, processed
// once under their own names before nvfp4_attn_input_shard_names.h renames anything.

#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_a4_tma_launch.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/attn_input_proj/nvfp4_shard/nvfp4_attn_input_shard.h"
#include "ops/linear/nvfp4/nvfp4_layout.h"

namespace ninfer::ops::detail {

// The shard's query|key|gate|value section output, in the parent's section order.
using Nvfp4AttnInputShardOutput = LinearBf16SegmentedOutput<3072, 512, 3072, 512>;

static_assert(kNvfp4AttnInputShardRows == 2 * (3072 + 512));
// Every row tile of the parent's routes (at most the 128-row A4 MMA and TMA tiles) divides each
// shard section, so a tile binds to one section and vector stores never straddle two.
static_assert((3072 % 128) == 0 && (512 % 128) == 0);

} // namespace ninfer::ops::detail
