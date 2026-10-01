#pragma once

// Tensor-parallel composition of the row-parallel projections and the vocabulary-split output
// head. `tp == nullptr` is the unsplit model and keeps the fused single-device calls. Planning
// uses the same workspace functions, so sizing and execution cannot disagree.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/tensor_parallel.h"
#include "models/qwen3_5/execution/parameters.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::models::qwen3_5::execution {

// residual += output(input), summing both ranks' partial products when split.
[[nodiscard]] std::size_t row_parallel_output_workspace_bytes(const LinearParameters& output,
                                                              std::int32_t first,
                                                              std::int32_t last, bool split);
void row_parallel_output(const Tensor& input, const LinearParameters& output, Tensor& residual,
                         const TensorParallelDeviceView* tp, WorkspaceArena& workspace,
                         cudaStream_t stream);

// residual += partial, summed over both ranks when split.
void row_parallel_residual(const Tensor& partial, Tensor& residual,
                           const TensorParallelDeviceView* tp, cudaStream_t stream);

// logits[:, t] = head(hidden[:, t]) over the whole vocabulary. A split head holds one rank's
// vocabulary rows; the two halves are gathered into `logits` on both ranks.
[[nodiscard]] std::size_t output_head_workspace_bytes(const LinearParameters& head,
                                                      std::int32_t first, std::int32_t last,
                                                      bool split);
void output_head(const Tensor& hidden, const LinearParameters& head, Tensor& logits,
                 const TensorParallelDeviceView* tp, WorkspaceArena& workspace,
                 cudaStream_t stream);

} // namespace ninfer::models::qwen3_5::execution
