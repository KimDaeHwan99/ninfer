#pragma once

// ninfer core - the two-rank tensor-parallel link.
//
// GeForce boards refuse peer access, so every cross-device payload crosses PCIe through host memory
// regardless of which CUDA entry point moves it. This link makes host memory the explicit
// transport: one portable, mapped, pinned host allocation holds, for each rank, a mailbox of
// per-CTA sequence numbers and a ring of staging slots. A collective kernel on rank r stores its
// contribution into its own staging slot, publishes the CTA's sequence number in its own mailbox,
// and polls the peer's mailbox before reading the peer's slot. No host event orders the two
// devices, so each rank enqueues (and graph-captures) its own work independently; the only
// cross-rank contract is that both ranks execute the same sequence of collectives.
//
// Sequence numbers live in device memory, one counter per collective CTA index, and are advanced
// by the collective kernels themselves. Eager execution and CUDA Graph replays therefore share one
// numbering, and a captured graph replays correctly without host-side patching.
//
// Ownership: rank r writes only its own mailbox and staging rings; it reads the peer's. Small
// payloads use a separate packed-line ring whose lines carry their own sequence tag.
//
// Eager callers may also run large collectives asynchronously on each rank's copy channel: two
// side streams move the payload with the copy engines (device -> host staging -> peer device) and
// combine it, so the issuing stream keeps computing. Host memory bandwidth bounds every transfer;
// the channel hides it behind compute rather than shortening it.

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer {

inline constexpr int kTensorParallelRanks = 2;
// CTAs of every collective (capped by the device's SM count). Collective CTAs wait on their peer
// counterpart, so all of them must be resident at once.
inline constexpr int kTensorParallelMaxBlocks = 32;
inline constexpr int kTensorParallelSlots     = 2;
// Bytes of one packed-line slot: 8-byte lines carrying 4 payload bytes and a sequence tag.
inline constexpr std::size_t kTensorParallelPackedSlotBytes = 256 * 1024;

// Host-mapped, written only by its owning rank.
struct TensorParallelMailbox {
    std::uint64_t published[kTensorParallelMaxBlocks];
};

// Pieces one asynchronous collective is split into so each rank's send and receive overlap.
inline constexpr int kTensorParallelCopyPieces = 4;
// Asynchronous collectives whose completion events stay distinct.
inline constexpr int kTensorParallelCopyTickets = 16;
// Rank-owned partial buffers callers may keep in flight across workspace scopes.
inline constexpr int kTensorParallelCopyPartials = 2;

// Host-mapped, written only by its owning rank: the call sequence each piece has reached host
// staging.
struct TensorParallelCopyMailbox {
    std::uint64_t published[kTensorParallelCopyPieces];
};

// One rank's copy-engine channel. Host-side state, mutated only by the thread that enqueues the
// rank's work; both ranks issue the same sequence of asynchronous collectives.
struct TensorParallelCopyChannel {
    cudaStream_t send    = nullptr; // device -> host staging, then publish
    cudaStream_t receive = nullptr; // wait for the peer, host staging -> scratch, combine
    cudaEvent_t fork     = nullptr;
    std::array<cudaEvent_t, kTensorParallelCopyTickets> sent{};
    std::array<cudaEvent_t, kTensorParallelCopyTickets> received{};
    std::uint64_t sequence = 0;
    // Two staging slots of `slot_bytes` each, used alternately by consecutive calls.
    std::byte* self_staging       = nullptr;
    const std::byte* peer_staging = nullptr;
    TensorParallelCopyMailbox* self_mailbox       = nullptr;
    const TensorParallelCopyMailbox* peer_mailbox = nullptr;
    // Device memory of this rank: the peer's payload lands in `scratch`; `partials` are free for
    // callers whose in-flight payload must outlive a workspace scope.
    std::byte* scratch = nullptr;
    std::array<std::byte*, kTensorParallelCopyPartials> partials{};
    std::uint64_t slot_bytes = 0;
    std::uint64_t timeout_ns = 0;
    std::int32_t rank        = 0;
};

// Plain view passed by value to collective kernels. All pointers are valid on the rank's device:
// mailboxes and staging are portable mapped host memory addressed through UVA, counters are
// device memory of this rank.
struct TensorParallelDeviceView {
    TensorParallelMailbox* self_mailbox       = nullptr;
    const TensorParallelMailbox* peer_mailbox = nullptr;
    std::byte* self_staging                   = nullptr;
    const std::byte* peer_staging             = nullptr;
    std::byte* self_packed                    = nullptr;
    const std::byte* peer_packed              = nullptr;
    std::uint64_t slot_bytes                  = 0;
    std::uint64_t* counters                   = nullptr;
    std::uint64_t timeout_ns                  = 0;
    std::int32_t rank                         = 0;
    std::int32_t max_blocks                   = 0;
    // Host-side; null when the link was built without copy channels.
    TensorParallelCopyChannel* copy = nullptr;
};

class TensorParallelLink {
public:
    // `devices[r]` is rank r's CUDA device. `slot_bytes` bounds the payload one rank contributes to
    // one collective. Creates the mapped host region, each rank's device counters and, with
    // `copy_channels`, each rank's copy channel (streams, events, scratch and partial buffers).
    TensorParallelLink(std::array<int, kTensorParallelRanks> devices, std::size_t slot_bytes,
                       bool copy_channels = true);
    ~TensorParallelLink();

    TensorParallelLink(const TensorParallelLink&)            = delete;
    TensorParallelLink& operator=(const TensorParallelLink&) = delete;
    TensorParallelLink(TensorParallelLink&&)                 = delete;
    TensorParallelLink& operator=(TensorParallelLink&&)      = delete;

    [[nodiscard]] TensorParallelDeviceView view(int rank) const;
    [[nodiscard]] int device(int rank) const { return devices_.at(static_cast<std::size_t>(rank)); }
    [[nodiscard]] std::size_t slot_bytes() const noexcept { return slot_bytes_; }
    [[nodiscard]] std::size_t host_bytes() const noexcept { return host_bytes_; }

private:
    std::array<int, kTensorParallelRanks> devices_{};
    std::size_t slot_bytes_ = 0;
    std::size_t host_bytes_ = 0;
    std::int32_t max_blocks_ = 0;
    std::uint64_t timeout_ns_ = 0;
    void* host_               = nullptr;
    std::array<std::uint64_t*, kTensorParallelRanks> counters_{};
    bool copy_channels_ = false;
    // Views hand each rank's thread mutable access to its own channel.
    mutable std::array<TensorParallelCopyChannel, kTensorParallelRanks> channels_{};
};

} // namespace ninfer
