#include "models/qwen3_5/execution/tensor_parallel.h"

#include "core/layout.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/tensor_parallel.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {

std::size_t row_parallel_output_workspace_bytes(const LinearParameters& output,
                                                std::int32_t first, std::int32_t last,
                                                bool split) {
    const auto& w = output.weight;
    if (!split) {
        return ops::linear_add_workspace_capacity_bytes(w.qtype, w.n, w.k, output.policy, first,
                                                        last);
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {w.n, last});
    (void)layout.alloc_bytes(
        ops::linear_workspace_capacity_bytes(w.qtype, w.n, w.k, output.policy, first, last));
    return layout.peak_bytes(1);
}

void row_parallel_output(const Tensor& input, const LinearParameters& output, Tensor& residual,
                         const TensorParallelDeviceView* tp, WorkspaceArena& workspace,
                         cudaStream_t stream) {
    if (tp == nullptr) {
        ops::linear_add(input, output.weight, residual, output.policy, workspace, stream);
        return;
    }
    auto scope    = workspace.scope();
    Tensor delta  = workspace.alloc(DType::BF16, {residual.ne[0], residual.ne[1]});
    {
        auto call = workspace.scope();
        ops::linear(input, output.weight, delta, output.policy, workspace, stream);
    }
    ops::tp_residual_allreduce(delta, residual, *tp, stream);
}

void row_parallel_residual(const Tensor& partial, Tensor& residual,
                           const TensorParallelDeviceView* tp, cudaStream_t stream) {
    if (tp == nullptr) {
        ops::residual_add(partial, residual, stream);
    } else {
        ops::tp_residual_allreduce(partial, residual, *tp, stream);
    }
}

std::size_t output_head_workspace_bytes(const LinearParameters& head, std::int32_t first,
                                        std::int32_t last, bool split) {
    const auto& w = head.weight;
    const auto scratch =
        ops::linear_workspace_capacity_bytes(w.qtype, w.n, w.k, head.policy, first, last);
    if (!split) { return scratch; }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {w.n, last});
    (void)layout.alloc_bytes(scratch);
    return layout.peak_bytes(1);
}

void output_head(const Tensor& hidden, const LinearParameters& head, Tensor& logits,
                 const TensorParallelDeviceView* tp, WorkspaceArena& workspace,
                 cudaStream_t stream) {
    if (tp == nullptr || head.weight.n == logits.ne[0]) {
        ops::linear(hidden, head.weight, logits, head.policy, workspace, stream);
        return;
    }
    if (static_cast<std::int64_t>(head.weight.n) * 2 != logits.ne[0]) {
        throw std::invalid_argument("split output head rows are not half of the logits rows");
    }
    auto scope   = workspace.scope();
    Tensor local = workspace.alloc(DType::BF16, {head.weight.n, logits.ne[1]});
    {
        auto call = workspace.scope();
        ops::linear(hidden, head.weight, local, head.policy, workspace, stream);
    }
    ops::tp_allgather_rows(local, logits, *tp, stream);
}

} // namespace ninfer::models::qwen3_5::execution
