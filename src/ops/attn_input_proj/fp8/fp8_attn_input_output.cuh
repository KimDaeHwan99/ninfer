#pragma once

#include "ops/linear/common/output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kFp8AttnInputQueryRows = 6144;
inline constexpr std::int32_t kFp8AttnInputKeyRows   = 1024;
inline constexpr std::int32_t kFp8AttnInputGateRows  = 6144;
inline constexpr std::int32_t kFp8AttnInputKeyBegin  = kFp8AttnInputQueryRows;
inline constexpr std::int32_t kFp8AttnInputGateBegin = kFp8AttnInputKeyBegin + kFp8AttnInputKeyRows;
inline constexpr std::int32_t kFp8AttnInputValueBegin =
    kFp8AttnInputGateBegin + kFp8AttnInputGateRows;

static_assert((kFp8AttnInputQueryRows % 8) == 0);
static_assert((kFp8AttnInputKeyRows % 8) == 0);
static_assert((kFp8AttnInputGateRows % 8) == 0);

using Fp8AttentionInputOutput = LinearBf16SegmentedOutput<6144, 1024, 6144, 1024>;
// One tensor-parallel rank's shard: half of the query and KV heads, in the same Q|K|G|V order.
using Fp8AttentionInputShardOutput = LinearBf16SegmentedOutput<3072, 512, 3072, 512>;

inline constexpr std::int32_t kFp8AttnInputRows      = 14336;
inline constexpr std::int32_t kFp8AttnInputShardRows = 7168;

// Calls `launch(output)` with the segmented output matching the parent's row profile.
template <class Launch>
void with_fp8_attention_output(std::int32_t parent_rows, void* q, void* k, void* gate, void* v,
                               Launch&& launch) {
    const auto pointers = [&]<class Output>() {
        return Output{static_cast<__nv_bfloat16*>(q), static_cast<__nv_bfloat16*>(k),
                      static_cast<__nv_bfloat16*>(gate), static_cast<__nv_bfloat16*>(v)};
    };
    if (parent_rows == kFp8AttnInputShardRows) {
        launch(pointers.template operator()<Fp8AttentionInputShardOutput>());
    } else {
        launch(pointers.template operator()<Fp8AttentionInputOutput>());
    }
}

} // namespace ninfer::ops::detail
