#pragma once

// ninfer::ops::detail - private launch prototypes for the tensor-parallel collectives.

#include "core/tensor.h"
#include "core/tensor_parallel.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Payload bytes above which tp_residual_allreduce_async uses the copy channel.
inline constexpr std::size_t kTensorParallelCopyMinBytes = std::size_t{1} << 20;

void tp_residual_allreduce_launch(const Tensor& partial, Tensor& residual,
                                  const TensorParallelDeviceView& tp, cudaStream_t stream);
std::uint64_t tp_residual_allreduce_copy_launch(const Tensor& partial, Tensor& residual,
                                                const TensorParallelDeviceView& tp,
                                                cudaStream_t stream);
void tp_wait_launch(const TensorParallelDeviceView& tp, std::uint64_t ticket, cudaStream_t stream);
void tp_allgather_rows_launch(const Tensor& local, Tensor& destination,
                              const TensorParallelDeviceView& tp, cudaStream_t stream);

} // namespace ninfer::ops::detail
