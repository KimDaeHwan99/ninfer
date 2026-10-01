#include "core/tensor_parallel.h"

#include "core/device.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

constexpr std::size_t kStagingAlignment = 4096;

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

std::uint64_t timeout_from_environment() {
    // A collective whose peer never arrives means the two ranks diverged; trap instead of hanging.
    std::uint64_t milliseconds = 120000;
    if (const char* value = std::getenv("NINFER_TP_TIMEOUT_MS"); value != nullptr) {
        milliseconds = std::strtoull(value, nullptr, 10);
        if (milliseconds == 0) {
            throw std::invalid_argument("NINFER_TP_TIMEOUT_MS must be a positive integer");
        }
    }
    return milliseconds * 1000000ULL;
}

class DeviceScope {
public:
    explicit DeviceScope(int device) {
        CUDA_CHECK(cudaGetDevice(&previous_));
        CUDA_CHECK(cudaSetDevice(device));
    }

    ~DeviceScope() { (void)cudaSetDevice(previous_); }

    DeviceScope(const DeviceScope&)            = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;

private:
    int previous_ = 0;
};

} // namespace

TensorParallelLink::TensorParallelLink(std::array<int, kTensorParallelRanks> devices,
                                       std::size_t slot_bytes, bool copy_channels)
    : devices_(devices), slot_bytes_(align_up(slot_bytes, kStagingAlignment)),
      timeout_ns_(timeout_from_environment()), copy_channels_(copy_channels) {
    if (devices_[0] == devices_[1]) {
        throw std::invalid_argument("tensor-parallel ranks must use distinct devices");
    }
    if (slot_bytes == 0) { throw std::invalid_argument("tensor-parallel slot bytes must be nonzero"); }
    int sm_count = kTensorParallelMaxBlocks;
    for (const int device : devices_) {
        cudaDeviceProp props{};
        CUDA_CHECK(cudaGetDeviceProperties(&props, device));
        if (!props.unifiedAddressing || !props.canMapHostMemory) {
            throw std::runtime_error("tensor-parallel link requires UVA and mapped host memory");
        }
        sm_count = std::min(sm_count, props.multiProcessorCount);
    }
    // Every collective CTA spins on its peer counterpart, so all of them must be resident at once.
    max_blocks_ = std::max(1, sm_count);

    const std::size_t mailboxes = align_up(sizeof(TensorParallelMailbox) * kTensorParallelRanks,
                                           kStagingAlignment);
    const std::size_t kernel_bytes =
        mailboxes + slot_bytes_ * kTensorParallelSlots * kTensorParallelRanks +
        kTensorParallelPackedSlotBytes * kTensorParallelSlots * kTensorParallelRanks;
    const std::size_t copy_mailboxes =
        align_up(sizeof(TensorParallelCopyMailbox) * kTensorParallelRanks, kStagingAlignment);
    host_bytes_ = kernel_bytes;
    if (copy_channels_) {
        host_bytes_ += copy_mailboxes + slot_bytes_ * kTensorParallelSlots * kTensorParallelRanks;
    }
    {
        DeviceScope scope(devices_[0]);
        CUDA_CHECK(cudaHostAlloc(&host_, host_bytes_, cudaHostAllocPortable | cudaHostAllocMapped));
    }
    // Packed lines must start with a tag no sequence uses.
    std::memset(host_, 0, host_bytes_);
    for (std::size_t rank = 0; rank < kTensorParallelRanks; ++rank) {
        DeviceScope scope(devices_[rank]);
        void* counters = nullptr;
        CUDA_CHECK(cudaMalloc(&counters, sizeof(std::uint64_t) * kTensorParallelMaxBlocks));
        CUDA_CHECK(cudaMemset(counters, 0, sizeof(std::uint64_t) * kTensorParallelMaxBlocks));
        CUDA_CHECK(cudaDeviceSynchronize());
        counters_[rank] = static_cast<std::uint64_t*>(counters);
        // Under UVA a portable mapped allocation has one address on every device.
        void* mapped = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&mapped, host_, 0));
        if (mapped != host_) {
            throw std::runtime_error("tensor-parallel host region is not identity-mapped");
        }
    }
    if (copy_channels_) {
        auto* base          = static_cast<std::byte*>(host_) + kernel_bytes;
        auto* copy_mailbox  = reinterpret_cast<TensorParallelCopyMailbox*>(base);
        auto* copy_staging  = base + copy_mailboxes;
        const auto staging  = [&](std::size_t r) {
            return copy_staging + r * slot_bytes_ * kTensorParallelSlots;
        };
        for (std::size_t rank = 0; rank < kTensorParallelRanks; ++rank) {
            DeviceScope scope(devices_[rank]);
            auto& channel = channels_[rank];
            int least     = 0;
            int greatest  = 0;
            CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
            // Combines and handshakes are short; let them take the next free SM slot.
            CUDA_CHECK(cudaStreamCreateWithPriority(&channel.send, cudaStreamNonBlocking, greatest));
            CUDA_CHECK(
                cudaStreamCreateWithPriority(&channel.receive, cudaStreamNonBlocking, greatest));
            CUDA_CHECK(cudaEventCreateWithFlags(&channel.fork, cudaEventDisableTiming));
            for (std::size_t t = 0; t < kTensorParallelCopyTickets; ++t) {
                CUDA_CHECK(cudaEventCreateWithFlags(&channel.sent[t], cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(&channel.received[t], cudaEventDisableTiming));
            }
            void* scratch = nullptr;
            CUDA_CHECK(cudaMalloc(&scratch, slot_bytes_ * (1 + kTensorParallelCopyPartials)));
            channel.scratch = static_cast<std::byte*>(scratch);
            for (std::size_t p = 0; p < kTensorParallelCopyPartials; ++p) {
                channel.partials[p] = channel.scratch + (1 + p) * slot_bytes_;
            }
            channel.self_staging = staging(rank);
            channel.peer_staging = staging(1 - rank);
            channel.self_mailbox = copy_mailbox + rank;
            channel.peer_mailbox = copy_mailbox + (1 - rank);
            channel.slot_bytes   = slot_bytes_;
            channel.timeout_ns   = timeout_ns_;
            channel.rank         = static_cast<std::int32_t>(rank);
        }
    }
}

TensorParallelLink::~TensorParallelLink() {
    for (std::size_t rank = 0; rank < kTensorParallelRanks; ++rank) {
        auto& channel = channels_[rank];
        if (channel.send != nullptr || channel.scratch != nullptr) {
            DeviceScope scope(devices_[rank]);
            if (channel.send != nullptr) { (void)cudaStreamSynchronize(channel.send); }
            if (channel.receive != nullptr) { (void)cudaStreamSynchronize(channel.receive); }
            for (std::size_t t = 0; t < kTensorParallelCopyTickets; ++t) {
                if (channel.sent[t] != nullptr) { (void)cudaEventDestroy(channel.sent[t]); }
                if (channel.received[t] != nullptr) { (void)cudaEventDestroy(channel.received[t]); }
            }
            if (channel.fork != nullptr) { (void)cudaEventDestroy(channel.fork); }
            if (channel.send != nullptr) { (void)cudaStreamDestroy(channel.send); }
            if (channel.receive != nullptr) { (void)cudaStreamDestroy(channel.receive); }
            if (channel.scratch != nullptr) { (void)cudaFree(channel.scratch); }
        }
        if (counters_[rank] != nullptr) {
            DeviceScope scope(devices_[rank]);
            (void)cudaFree(counters_[rank]);
        }
    }
    if (host_ != nullptr) { (void)cudaFreeHost(host_); }
}

TensorParallelDeviceView TensorParallelLink::view(int rank) const {
    if (rank < 0 || rank >= kTensorParallelRanks) {
        throw std::out_of_range("tensor-parallel rank is out of range");
    }
    auto* base           = static_cast<std::byte*>(host_);
    auto* mailboxes      = reinterpret_cast<TensorParallelMailbox*>(base);
    const std::size_t mb = align_up(sizeof(TensorParallelMailbox) * kTensorParallelRanks,
                                    kStagingAlignment);
    const auto ring      = [&](int r) {
        return base + mb + static_cast<std::size_t>(r) * slot_bytes_ * kTensorParallelSlots;
    };
    const auto packed = [&](int r) {
        return base + mb + slot_bytes_ * kTensorParallelSlots * kTensorParallelRanks +
               static_cast<std::size_t>(r) * kTensorParallelPackedSlotBytes * kTensorParallelSlots;
    };
    const int peer = 1 - rank;
    return TensorParallelDeviceView{.self_mailbox = mailboxes + rank,
                                    .peer_mailbox = mailboxes + peer,
                                    .self_staging = ring(rank),
                                    .peer_staging = ring(peer),
                                    .self_packed  = packed(rank),
                                    .peer_packed  = packed(peer),
                                    .slot_bytes   = slot_bytes_,
                                    .counters     = counters_[static_cast<std::size_t>(rank)],
                                    .timeout_ns   = timeout_ns_,
                                    .rank         = rank,
                                    .max_blocks   = max_blocks_,
                                    .copy         = copy_channels_
                                                        ? &channels_[static_cast<std::size_t>(rank)]
                                                        : nullptr};
}

} // namespace ninfer
