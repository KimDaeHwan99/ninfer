#pragma once

#include "core/tensor_parallel.h"
#include "models/qwen3_5/execution/parameters.h"

namespace ninfer::models::qwen3_5::execution {

// `split` / `tp` select the tensor-parallel down projection, whose partial products are summed
// over both ranks into the residual.
[[nodiscard]] std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                              std::int32_t last, bool mtp = false,
                                              bool split = false);
void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, cudaStream_t stream,
         bool mtp = false, const TensorParallelDeviceView* tp = nullptr);

// delta = down(silu(gate(hidden)) * up(hidden)) of a dense FFN, composed from Linear calls: the
// MTP layer and a tensor-parallel rank's partial product. Fits ffn_workspace_bytes(split=true).
void dense_ffn_product(const Tensor& hidden, const DenseParameters& parameters, Tensor& delta,
                       WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::models::qwen3_5::execution
