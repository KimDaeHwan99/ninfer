#pragma once

// Tensor-parallel composition of the row-parallel projections and the vocabulary-split output
// head. `tp == nullptr` is the unsplit model and keeps the fused single-device calls. Planning
// uses the same workspace functions, so sizing and execution cannot disagree.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/tensor_parallel.h"
#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/tensor_parallel.h"

#include <cuda_runtime.h>

#include <array>
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

// Column slices of a prefill-width row-parallel stage. Each slice's all-reduce runs on the link's
// copy channel while the stream computes the next slice; `count == 0` keeps the synchronous
// stage. Slices only split independent token columns, so results equal the unsliced stage.
inline constexpr int kTensorParallelMaxSlices = 4;
struct TensorParallelSlices {
    std::int32_t count = 0;
    std::int32_t width = 0; // columns of every slice but the last, which takes the remainder
    std::array<ops::TensorParallelTicket, kTensorParallelMaxSlices> tickets{};

    [[nodiscard]] std::int32_t first(int slice) const { return slice * width; }
    [[nodiscard]] std::int32_t columns(int slice, std::int32_t total) const {
        return slice + 1 < count ? width : total - slice * width;
    }
};

// Slices for a `columns`-wide stage on `stream`, or none when unsplit, capturing, or too narrow
// for the copy channel to pay off.
[[nodiscard]] TensorParallelSlices tensor_parallel_slices(const TensorParallelDeviceView* tp,
                                                          std::int32_t columns,
                                                          cudaStream_t stream);

// residual += output(input) over `slices`, the products written to the channel's partial buffer
// `partial` so the in-flight all-reduces outlive the caller's workspace scope. Fills
// `slices.tickets`; the consumer waits each before touching that residual slice.
void row_parallel_output_sliced(const Tensor& input, const LinearParameters& output,
                                Tensor& residual, const TensorParallelDeviceView& tp, int partial,
                                TensorParallelSlices& slices, WorkspaceArena& workspace,
                                cudaStream_t stream);

// The channel's partial buffer `index` viewed as BF16 [rows, columns].
[[nodiscard]] Tensor tensor_parallel_partial(const TensorParallelDeviceView& tp, int index,
                                             std::int32_t rows, std::int32_t columns);

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
