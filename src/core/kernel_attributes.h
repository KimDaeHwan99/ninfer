#pragma once

#include <cuda_runtime.h>

#include <array>
#include <atomic>

namespace ninfer {

inline constexpr int kKernelAttributeDevices = 64;

// A kernel's maximum dynamic shared memory is an attribute of each device context, so a process
// launching the kernel on two devices must apply it on both. Applies `bytes` to `Kernel` once per
// device the calling thread launches it on; later calls cost one cudaGetDevice. Returns the
// attribute status on the current device.
template <auto Kernel>
cudaError_t set_kernel_max_dynamic_shared_memory(int bytes) {
    static std::array<std::atomic<int>, kKernelAttributeDevices> applied{};
    int device = 0;
    if (const cudaError_t status = cudaGetDevice(&device); status != cudaSuccess) {
        return status;
    }
    if (device < 0 || device >= kKernelAttributeDevices) {
        return cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    }
    auto& slot = applied[static_cast<std::size_t>(device)];
    if (slot.load(std::memory_order_acquire) >= bytes) { return cudaSuccess; }
    const cudaError_t status =
        cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    if (status == cudaSuccess) {
        int previous = slot.load(std::memory_order_relaxed);
        while (previous < bytes &&
               !slot.compare_exchange_weak(previous, bytes, std::memory_order_release)) {}
    }
    return status;
}

} // namespace ninfer
