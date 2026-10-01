// ninfer::ops - tensor-parallel collective wrappers: validate operands and the link view, then
// dispatch. Host-compiled; never includes the kernel header.
#include "ninfer/ops/tensor_parallel.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/launcher/tensor_parallel.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_link(const TensorParallelDeviceView& tp, const char* op) {
    if (tp.self_mailbox == nullptr || tp.peer_mailbox == nullptr || tp.self_staging == nullptr ||
        tp.peer_staging == nullptr || tp.self_packed == nullptr || tp.peer_packed == nullptr ||
        tp.counters == nullptr || tp.slot_bytes == 0 ||
        tp.max_blocks <= 0 || tp.max_blocks > kTensorParallelMaxBlocks ||
        (tp.rank != 0 && tp.rank != 1)) {
        throw std::invalid_argument(std::string(op) + ": invalid tensor-parallel link view");
    }
}

void require_bf16_packs(const Tensor& t, const char* op, const char* label) {
    if (t.dtype != DType::BF16 || !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + label +
                                    " must be contiguous non-null BF16");
    }
    if ((reinterpret_cast<std::uintptr_t>(t.data) & 15U) != 0 || t.numel() % 8 != 0) {
        throw std::invalid_argument(std::string(op) + ": " + label +
                                    " must be 16-byte aligned whole BF16x8 packs");
    }
}

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto a0 = reinterpret_cast<std::uintptr_t>(a.data);
    const auto b0 = reinterpret_cast<std::uintptr_t>(b.data);
    return a0 < b0 + b.bytes() && b0 < a0 + a.bytes();
}

void require_operands(const Tensor& partial, const Tensor& residual, const char* op) {
    require_bf16_packs(partial, op, "partial");
    require_bf16_packs(residual, op, "residual");
    for (int d = 0; d < 4; ++d) {
        if (partial.ne[d] != residual.ne[d]) {
            throw std::invalid_argument(std::string(op) + ": partial/residual shapes differ");
        }
    }
    if (overlaps(partial, residual)) {
        throw std::invalid_argument(std::string(op) + ": partial and residual overlap");
    }
}

} // namespace

void tp_residual_allreduce(const Tensor& partial, Tensor& residual,
                           const TensorParallelDeviceView& tp, cudaStream_t stream) {
    constexpr const char* op = "tp_residual_allreduce";
    require_link(tp, op);
    require_operands(partial, residual, op);
    if (partial.bytes() > tp.slot_bytes) {
        throw std::invalid_argument("tp_residual_allreduce: payload exceeds the link slot");
    }
    detail::tp_residual_allreduce_launch(partial, residual, tp, stream);
}

TensorParallelTicket tp_residual_allreduce_async(const Tensor& partial, Tensor& residual,
                                                 const TensorParallelDeviceView& tp,
                                                 cudaStream_t stream) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (tp.copy == nullptr || capture != cudaStreamCaptureStatusNone ||
        partial.bytes() <= detail::kTensorParallelCopyMinBytes) {
        tp_residual_allreduce(partial, residual, tp, stream);
        return 0;
    }
    constexpr const char* op = "tp_residual_allreduce_async";
    require_link(tp, op);
    require_operands(partial, residual, op);
    if (partial.bytes() > tp.copy->slot_bytes) {
        throw std::invalid_argument("tp_residual_allreduce_async: payload exceeds the link slot");
    }
    return detail::tp_residual_allreduce_copy_launch(partial, residual, tp, stream);
}

void tp_wait(const TensorParallelDeviceView& tp, TensorParallelTicket ticket, cudaStream_t stream) {
    if (ticket == 0) { return; }
    if (tp.copy == nullptr || ticket > tp.copy->sequence) {
        throw std::invalid_argument("tp_wait: ticket was not issued by this rank's channel");
    }
    detail::tp_wait_launch(tp, ticket, stream);
}

void tp_allgather_rows(const Tensor& local, Tensor& destination, const TensorParallelDeviceView& tp,
                       cudaStream_t stream) {
    constexpr const char* op = "tp_allgather_rows";
    require_link(tp, op);
    require_bf16_packs(local, op, "local");
    require_bf16_packs(destination, op, "destination");
    if (local.ne[2] != 1 || local.ne[3] != 1 || destination.ne[2] != 1 || destination.ne[3] != 1 ||
        destination.ne[0] != 2 * local.ne[0] || destination.ne[1] != local.ne[1] ||
        local.ne[0] % 8 != 0) {
        throw std::invalid_argument("tp_allgather_rows: expected [R,C] -> [2R,C] with R % 8 == 0");
    }
    if (overlaps(local, destination)) {
        throw std::invalid_argument("tp_allgather_rows: local and destination overlap");
    }
    if (local.bytes() > tp.slot_bytes) {
        throw std::invalid_argument("tp_allgather_rows: payload exceeds the link slot");
    }
    detail::tp_allgather_rows_launch(local, destination, tp, stream);
}

} // namespace ninfer::ops
