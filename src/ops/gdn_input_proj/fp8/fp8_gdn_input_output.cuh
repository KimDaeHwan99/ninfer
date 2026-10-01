#pragma once
#include "ops/linear/common/output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
using Fp8GdnInputOutput = LinearBf16SegmentedOutput<10240, 6144>;
// One tensor-parallel rank's shard: half of the key and value heads, in the same QKV | Z order.
using Fp8GdnInputShardOutput = LinearBf16SegmentedOutput<5120, 3072>;

inline constexpr std::int32_t kFp8GdnInputShardRows = 8192;

// Calls `launch(output)` with the segmented output matching the parent's row profile.
template <class Launch>
void with_fp8_gdn_output(std::int32_t parent_rows, void* qkv, void* z, Launch&& launch) {
    if (parent_rows == kFp8GdnInputShardRows) {
        launch(Fp8GdnInputShardOutput{static_cast<__nv_bfloat16*>(qkv),
                                      static_cast<__nv_bfloat16*>(z)});
    } else {
        launch(Fp8GdnInputOutput{static_cast<__nv_bfloat16*>(qkv), static_cast<__nv_bfloat16*>(z)});
    }
}
} // namespace ninfer::ops::detail
