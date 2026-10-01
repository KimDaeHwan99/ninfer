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
                                       std::size_t slot_bytes)
    : devices_(devices), slot_bytes_(align_up(slot_bytes, kStagingAlignment)),
      timeout_ns_(timeout_from_environment()) {
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
    host_bytes_ = mailboxes + slot_bytes_ * kTensorParallelSlots * kTensorParallelRanks;
    {
        DeviceScope scope(devices_[0]);
        CUDA_CHECK(cudaHostAlloc(&host_, host_bytes_, cudaHostAllocPortable | cudaHostAllocMapped));
    }
    std::memset(host_, 0, mailboxes);
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
}

TensorParallelLink::~TensorParallelLink() {
    for (std::size_t rank = 0; rank < kTensorParallelRanks; ++rank) {
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
    const int peer = 1 - rank;
    return TensorParallelDeviceView{.self_mailbox = mailboxes + rank,
                                    .peer_mailbox = mailboxes + peer,
                                    .self_staging = ring(rank),
                                    .peer_staging = ring(peer),
                                    .slot_bytes   = slot_bytes_,
                                    .counters     = counters_[static_cast<std::size_t>(rank)],
                                    .timeout_ns   = timeout_ns_,
                                    .rank         = rank,
                                    .max_blocks   = max_blocks_};
}

} // namespace ninfer
