// Implements: include/ninfer/ops/tensor_parallel.h
#include "ops/launcher/tensor_parallel.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/tensor_parallel.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// Every collective launches all `max_blocks` CTAs so the per-CTA sequences stay one global call
// sequence; CTAs past the payload only take part in the exchange.
std::int64_t packs_per_block(std::int64_t packs, const TensorParallelDeviceView& tp) {
    return (packs + tp.max_blocks - 1) / tp.max_blocks;
}

} // namespace

void tp_residual_allreduce_launch(const Tensor& partial, Tensor& residual,
                                  const TensorParallelDeviceView& tp, cudaStream_t stream) {
    const std::int64_t packs = residual.numel() / 8;
    if (residual.bytes() <= static_cast<std::size_t>(kTensorParallelPackedMaxBytes)) {
        tp_residual_allreduce_packed_kernel<<<tp.max_blocks, kTensorParallelThreads, 0, stream>>>(
            static_cast<const std::uint32_t*>(partial.data),
            static_cast<std::uint32_t*>(residual.data), packs * 4, tp);
    } else {
        tp_residual_allreduce_kernel<<<tp.max_blocks, kTensorParallelThreads, 0, stream>>>(
            static_cast<const uint4*>(partial.data), static_cast<uint4*>(residual.data), packs,
            packs_per_block(packs, tp), tp);
    }
    CUDA_CHECK(cudaGetLastError());
}

void tp_allgather_rows_launch(const Tensor& local, Tensor& destination,
                              const TensorParallelDeviceView& tp, cudaStream_t stream) {
    const std::int64_t packs     = local.numel() / 8;
    const std::int64_t row_packs = local.ne[0] / 8;
    tp_allgather_rows_kernel<<<tp.max_blocks, kTensorParallelThreads, 0, stream>>>(
        static_cast<const uint4*>(local.data), static_cast<uint4*>(destination.data), packs,
        row_packs, packs_per_block(packs, tp), tp);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
