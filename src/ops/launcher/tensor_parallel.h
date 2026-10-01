#pragma once

// ninfer::ops::detail - private launch prototypes for the tensor-parallel collectives.

#include "core/tensor.h"
#include "core/tensor_parallel.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void tp_residual_allreduce_launch(const Tensor& partial, Tensor& residual,
                                  const TensorParallelDeviceView& tp, cudaStream_t stream);
void tp_allgather_rows_launch(const Tensor& local, Tensor& destination,
                              const TensorParallelDeviceView& tp, cudaStream_t stream);

} // namespace ninfer::ops::detail
