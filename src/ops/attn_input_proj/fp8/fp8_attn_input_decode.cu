#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"

#include "core/device.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/fp8/fp8_a16_gemv.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

void fp8_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, cudaStream_t stream) {
    using Geometry = Fp8N14336K5120;
    // Keep six CTAs resident with the four-segment output (39 registers, no spills).
    using Schedule = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 6>;
    with_fp8_attention_output(weight.n, q.data, k.data, gate.data, v.data, [&](const auto& output) {
        launch_fp8_a16_gemv<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
            fp8_a16_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    });
}

} // namespace ninfer::ops::detail
