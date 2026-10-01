// Implements: include/ninfer/ops/tensor_parallel.h
#include "ops/launcher/tensor_parallel.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/tensor_parallel.cuh"

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// One pack per thread per CTA keeps small (decode) payloads latency-bound on the fewest PCIe
// round trips; larger payloads loop inside at most `max_blocks` resident CTAs.
struct Partition {
    int blocks                  = 1;
    std::int64_t packs_per_block = 0;
};

Partition partition(std::int64_t packs, const TensorParallelDeviceView& tp) {
    const std::int64_t wanted = (packs + kTensorParallelThreads - 1) / kTensorParallelThreads;
    Partition out;
    out.blocks          = static_cast<int>(std::clamp<std::int64_t>(wanted, 1, tp.max_blocks));
    out.packs_per_block = (packs + out.blocks - 1) / out.blocks;
    return out;
}

} // namespace

void tp_residual_allreduce_launch(const Tensor& partial, Tensor& residual,
                                  const TensorParallelDeviceView& tp, cudaStream_t stream) {
    const std::int64_t packs = residual.numel() / 8;
    const auto split         = partition(packs, tp);
    tp_residual_allreduce_kernel<<<split.blocks, kTensorParallelThreads, 0, stream>>>(
        static_cast<const uint4*>(partial.data), static_cast<uint4*>(residual.data), packs,
        split.packs_per_block, tp);
    CUDA_CHECK(cudaGetLastError());
}

void tp_allgather_rows_launch(const Tensor& local, Tensor& destination,
                              const TensorParallelDeviceView& tp, cudaStream_t stream) {
    const std::int64_t packs     = local.numel() / 8;
    const std::int64_t row_packs = local.ne[0] / 8;
    const auto split             = partition(packs, tp);
    tp_allgather_rows_kernel<<<split.blocks, kTensorParallelThreads, 0, stream>>>(
        static_cast<const uint4*>(local.data), static_cast<uint4*>(destination.data), packs,
        row_packs, split.packs_per_block, tp);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
