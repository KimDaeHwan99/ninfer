// Implements: include/ninfer/ops/tensor_parallel.h
#include "ops/launcher/tensor_parallel.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/tensor_parallel.cuh"

#include <algorithm>
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

std::uint64_t tp_residual_allreduce_copy_launch(const Tensor& partial, Tensor& residual,
                                                const TensorParallelDeviceView& tp,
                                                cudaStream_t stream) {
    auto& channel             = *tp.copy;
    const std::size_t bytes   = partial.bytes();
    const std::uint64_t ticket = ++channel.sequence;
    const std::size_t slot    = (ticket % kTensorParallelSlots) * channel.slot_bytes;
    // Pieces of at least 1 MiB keep per-copy overhead small while the send of piece i+1 overlaps
    // the receive of piece i.
    const std::size_t pieces = std::clamp<std::size_t>(bytes >> 20, 1, kTensorParallelCopyPieces);
    const std::size_t piece  = (bytes / pieces + 15) / 16 * 16;
    const auto& previous     = channel.received[(ticket - 1) % kTensorParallelCopyTickets];

    CUDA_CHECK(cudaEventRecord(channel.fork, stream));
    CUDA_CHECK(cudaStreamWaitEvent(channel.send, channel.fork, 0));
    // Staging slot (ticket % 2) was last read by the peer for ticket - 2. The peer published
    // ticket - 1 only after that read (its send stream waits on its previous receive), and this
    // rank's previous receive observed that publication.
    if (ticket > 1) { CUDA_CHECK(cudaStreamWaitEvent(channel.send, previous, 0)); }
    CUDA_CHECK(cudaStreamWaitEvent(channel.receive, channel.fork, 0));
    const auto* own = static_cast<const std::byte*>(partial.data);
    auto* out       = static_cast<std::byte*>(residual.data);
    for (std::size_t p = 0, offset = 0; offset < bytes; ++p, offset += piece) {
        const std::size_t count = std::min(piece, bytes - offset);
        CUDA_CHECK(cudaMemcpyAsync(channel.self_staging + slot + offset, own + offset, count,
                                   cudaMemcpyDeviceToHost, channel.send));
        tp_copy_publish_kernel<<<1, 1, 0, channel.send>>>(&channel.self_mailbox->published[p],
                                                          ticket);
        CUDA_CHECK(cudaGetLastError());
    }
    for (std::size_t p = 0, offset = 0; offset < bytes; ++p, offset += piece) {
        const std::size_t count = std::min(piece, bytes - offset);
        tp_copy_wait_kernel<<<1, 1, 0, channel.receive>>>(&channel.peer_mailbox->published[p],
                                                          ticket, channel.timeout_ns);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(channel.scratch + offset, channel.peer_staging + slot + offset,
                                   count, cudaMemcpyHostToDevice, channel.receive));
        const auto packs  = static_cast<std::int64_t>(count / 16);
        const auto blocks = static_cast<int>(
            std::min<std::int64_t>((packs + kTensorParallelThreads - 1) / kTensorParallelThreads,
                                   4 * tp.max_blocks));
        tp_copy_combine_kernel<<<blocks, kTensorParallelThreads, 0, channel.receive>>>(
            reinterpret_cast<const uint4*>(own + offset),
            reinterpret_cast<const uint4*>(channel.scratch + offset),
            reinterpret_cast<uint4*>(out + offset), packs, channel.rank);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(channel.sent[ticket % kTensorParallelCopyTickets], channel.send));
    CUDA_CHECK(
        cudaEventRecord(channel.received[ticket % kTensorParallelCopyTickets], channel.receive));
    return ticket;
}

void tp_wait_launch(const TensorParallelDeviceView& tp, std::uint64_t ticket, cudaStream_t stream) {
    // A later call re-recording the slot only lengthens the wait: both side streams run in order.
    const auto index = ticket % kTensorParallelCopyTickets;
    CUDA_CHECK(cudaStreamWaitEvent(stream, tp.copy->sent[index], 0));
    CUDA_CHECK(cudaStreamWaitEvent(stream, tp.copy->received[index], 0));
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
