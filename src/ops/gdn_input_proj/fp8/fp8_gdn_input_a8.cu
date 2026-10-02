#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh"
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "ops/linear/fp8/fp8_instances.cuh"

namespace ninfer::ops::detail {
namespace {
using Tma64x128  = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma192x128 = Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>;
using MidBulk = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 2, 1>, 170, 4, 8>;
using Bulk    = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;
// A tensor-parallel rank's 8192-row shard balances its final wave for an RTX 5060 Ti's 36 SMs
// through 2048 tokens (T=512 287 -> 268 us, T=1536 831 -> 768 us, T=2048 1071 -> 1044 us); at 3000
// tokens the 170-CTA plan is 3% faster.
using ShardMidBulk = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 2, 1>, 36, 4, 8>;
using ShardBulk    = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 36, 4, 8>;

} // namespace

std::size_t fp8_gdn_input_partial_capacity_bytes(std::int32_t max_tokens) {
    // Every token count routed to a split-K schedule may split its final wave: a tensor-parallel
    // shard's half of the rows leaves that wave underfilled already above 192 tokens.
    return max_tokens > 192 ? Bulk::kPartialBytes : 0;
}

void fp8_gdn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                             Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const auto operands = fp8_a8_operands(weight, workspace, x.ne[1]);
    with_fp8_gdn_output(weight.n, qkv.data, z.data, [&](const auto& output) {
        const auto launch = [&]<class Schedule>() {
            using S = Fp8ScheduleInstance<Schedule, 5120>;
            if constexpr (S::kTmaSwizzle)
                launch_fp8_a8_tma_mma<S>(operands, output, LinearIdentityEpilogue{}, stream,
                                         workspace.partials);
            else
                launch_fp8_a8_mma<S>(operands, output, LinearIdentityEpilogue{}, stream);
        };
        if (x.ne[1] <= 32) return launch.template operator()<Fp8A8T32R32K128>();
        if (x.ne[1] <= 64) return launch.template operator()<Fp8A8T64R128K256>();
        if (x.ne[1] <= 128) return launch.template operator()<Tma64x128>();
        if (x.ne[1] <= 192) return launch.template operator()<Tma192x128>();
        // Smaller output tiles leave only two full-K tiles to split near the 512-token anchor.
        const bool shard = weight.n == 8192 && x.ne[1] <= 2048;
        if (x.ne[1] > 384 && x.ne[1] <= 512)
            return shard ? launch.template operator()<ShardMidBulk>() : launch.template operator()<MidBulk>();
        shard ? launch.template operator()<ShardBulk>() : launch.template operator()<Bulk>();
    });
}
} // namespace ninfer::ops::detail
