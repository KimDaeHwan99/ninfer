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
// Ownership: rank r writes only its own mailbox and staging ring; it reads the peer's. A slot is
// reused only after the peer has acknowledged consuming that slot's previous payload, so
// back-to-back collectives need no host synchronization.

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer {

inline constexpr int kTensorParallelRanks = 2;
// Upper bound on the CTAs of one collective. Collective CTAs wait on their peer counterpart, so a
// launch never exceeds the device's resident capacity (see TensorParallelLink::blocks()).
inline constexpr int kTensorParallelMaxBlocks = 32;
inline constexpr int kTensorParallelSlots     = 2;

// Host-mapped, written only by its owning rank.
struct TensorParallelMailbox {
    std::uint64_t published[kTensorParallelMaxBlocks];
    std::uint64_t consumed[kTensorParallelMaxBlocks];
};

// Plain view passed by value to collective kernels. All pointers are valid on the rank's device:
// mailboxes and staging are portable mapped host memory addressed through UVA, counters are
// device memory of this rank.
struct TensorParallelDeviceView {
    TensorParallelMailbox* self_mailbox       = nullptr;
    const TensorParallelMailbox* peer_mailbox = nullptr;
    std::byte* self_staging                   = nullptr;
    const std::byte* peer_staging             = nullptr;
    std::uint64_t slot_bytes                  = 0;
    std::uint64_t* counters                   = nullptr;
    std::uint64_t timeout_ns                  = 0;
    std::int32_t rank                         = 0;
    std::int32_t max_blocks                   = 0;
};

class TensorParallelLink {
public:
    // `devices[r]` is rank r's CUDA device. `slot_bytes` bounds the payload one rank contributes to
    // one collective. Creates the mapped host region and each rank's device counters.
    TensorParallelLink(std::array<int, kTensorParallelRanks> devices, std::size_t slot_bytes);
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
};

} // namespace ninfer
