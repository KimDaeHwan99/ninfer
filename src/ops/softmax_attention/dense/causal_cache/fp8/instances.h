#pragma once

#include "ops/softmax_attention/dense/causal_cache/fp8/schedule.cuh"

namespace ninfer::ops::detail {

template <class G, int Tokens>
struct Fp8KvGroupedInstance {
    static_assert(Tokens > 0 && Tokens * G::GroupSize <= 64);
    static constexpr int kRowTiles = (Tokens * G::GroupSize + 15) / 16;
    using Schedule                 = Fp8KvGroupedMmaSchedule<Tokens,
                                             kRowTiles == 4   ? 16
                                                             : kRowTiles == 3 ? 12
                                                                              : 8,
                                             Tokens == 1 ? 32 : 64, Tokens == 1 ? 2 : 1>;
    using Merge                    = Fp8KvMergeSchedule<G::QHeads == 24 ? 256 : 64>;
};

// FP8 prefill tiles. With FP16-accumulated PV, 128x64 tiles beat 64x32 on an RTX 5060 Ti (T=2048 at
// 16K keys 5.22 vs 5.85 ms).
inline constexpr int kFp8TiledQueryRows = 128;
using Fp8KvTiledInstance                = Fp8KvTiledMmaSchedule<kFp8TiledQueryRows, 64>;

} // namespace ninfer::ops::detail
