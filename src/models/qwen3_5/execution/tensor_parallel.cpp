#include "models/qwen3_5/execution/tensor_parallel.h"

#include "core/device.h" // CUDA_CHECK
#include "core/layout.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/tensor_parallel.h"

#include <algorithm>
#include <cstdint>
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

TensorParallelSlices tensor_parallel_slices(const TensorParallelDeviceView* tp,
                                            std::int32_t columns, cudaStream_t stream) {
    // Below 512 columns a slice's payload nears the copy channel's minimum and its GEMMs lose
    // efficiency; up to four slices leave one slice's all-reduce exposed.
    constexpr std::int32_t kMinColumns = 512;
    constexpr std::int32_t kAlign      = 16;
    TensorParallelSlices slices;
    if (tp == nullptr || tp->copy == nullptr || columns < kMinColumns) { return slices; }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) { return slices; }
    const std::int32_t count = std::clamp(columns / kMinColumns, 2, kTensorParallelMaxSlices);
    slices.width = (columns / count + kAlign - 1) / kAlign * kAlign;
    slices.count = (columns + slices.width - 1) / slices.width;
    return slices;
}

Tensor tensor_parallel_partial(const TensorParallelDeviceView& tp, int index, std::int32_t rows,
                               std::int32_t columns) {
    if (tp.copy == nullptr || index < 0 || index >= kTensorParallelCopyPartials) {
        throw std::invalid_argument("tensor-parallel partial buffer is unavailable");
    }
    if (rows <= 0 || columns <= 0 ||
        static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(columns) *
                sizeof(std::uint16_t) >
            tp.copy->slot_bytes) {
        throw std::invalid_argument("tensor-parallel partial exceeds the link slot");
    }
    return Tensor(tp.copy->partials[static_cast<std::size_t>(index)], DType::BF16,
                  {rows, columns});
}

void row_parallel_output_sliced(const Tensor& input, const LinearParameters& output,
                                Tensor& residual, const TensorParallelDeviceView& tp, int partial,
                                TensorParallelSlices& slices, WorkspaceArena& workspace,
                                cudaStream_t stream) {
    const auto total = static_cast<std::int32_t>(residual.ne[1]);
    Tensor products  = tensor_parallel_partial(tp, partial, residual.ne[0], total);
    for (int i = 0; i < slices.count; ++i) {
        const auto first   = slices.first(i);
        const auto columns = slices.columns(i, total);
        Tensor product     = products.slice(1, first, columns);
        {
            auto call = workspace.scope();
            ops::linear(input.slice(1, first, columns), output.weight, product, output.policy,
                        workspace, stream);
        }
        Tensor target     = residual.slice(1, first, columns);
        slices.tickets[i] = ops::tp_residual_allreduce_async(product, target, tp, stream);
    }
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
    // Gather in column slices that fit one link slot (scoring tiles exceed it).
    const auto row_bytes = static_cast<std::uint64_t>(head.weight.n) * sizeof(std::uint16_t);
    const auto slice     = static_cast<std::int32_t>(
        std::min<std::uint64_t>(logits.ne[1], std::max<std::uint64_t>(1, tp->slot_bytes / row_bytes)));
    auto scope   = workspace.scope();
    Tensor local = workspace.alloc(DType::BF16, {head.weight.n, logits.ne[1]});
    {
        auto call = workspace.scope();
        ops::linear(hidden, head.weight, local, head.policy, workspace, stream);
    }
    for (std::int32_t first = 0; first < logits.ne[1]; first += slice) {
        const std::int32_t count = std::min(slice, logits.ne[1] - first);
        Tensor part              = local.slice(1, first, count);
        Tensor destination       = logits.slice(1, first, count);
        ops::tp_allgather_rows(part, destination, *tp, stream);
    }
}

} // namespace ninfer::models::qwen3_5::execution
