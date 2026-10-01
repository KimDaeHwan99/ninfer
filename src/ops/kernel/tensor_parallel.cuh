#pragma once

// Implements: include/ninfer/ops/tensor_parallel.h
//
// Every collective launches exactly `max_blocks` CTAs, so each CTA's device-resident counter
// advances once per collective and all CTAs of both ranks agree on one call sequence. Slot
// `sequence % 2` is therefore reused two calls later, and a rank only reaches call c after it
// observed every peer CTA in call c-1; the peer's call c-1 kernel started only after its call c-2
// kernel (the last reader of that slot) completed, so slots need no release acknowledgement.
//
// Small payloads use packed lines: each 8-byte word carries 4 payload bytes and the low 32 bits
// of the sequence, so a reader polls the data itself and one PCIe round trip delivers both. Large
// payloads stage plain bytes, then publish a per-CTA flag after a system fence.

#include "core/tensor_parallel.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>

namespace ninfer::ops {

inline constexpr int kTensorParallelThreads = 256;
// Payload bytes per rank at or below which the packed-line protocol is used.
inline constexpr std::int64_t kTensorParallelPackedMaxBytes =
    static_cast<std::int64_t>(kTensorParallelPackedSlotBytes / 2);

__device__ __forceinline__ std::uint64_t tp_load_acquire(const std::uint64_t* address) {
    std::uint64_t value;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ void tp_store_release(std::uint64_t* address, std::uint64_t value) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(address), "l"(value) : "memory");
}

__device__ __forceinline__ std::uint64_t tp_load_volatile(const std::uint64_t* address) {
    std::uint64_t value;
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ void tp_store_volatile(std::uint64_t* address, std::uint64_t value) {
    asm volatile("st.volatile.global.u64 [%0], %1;" ::"l"(address), "l"(value) : "memory");
}

__device__ __forceinline__ std::uint64_t tp_global_timer() {
    std::uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

[[noreturn]] __device__ __forceinline__ void tp_timeout(const char* what, std::uint64_t target) {
    printf("ninfer tensor-parallel: timed out waiting for peer %s (CTA %d, target %llu)\n", what,
           static_cast<int>(blockIdx.x), static_cast<unsigned long long>(target));
    __trap();
}

__device__ __forceinline__ void tp_wait_at_least(const std::uint64_t* address, std::uint64_t target,
                                                 std::uint64_t timeout_ns, const char* what) {
    if (tp_load_acquire(address) >= target) { return; }
    const std::uint64_t start = tp_global_timer();
    while (tp_load_acquire(address) < target) {
        if (tp_global_timer() - start > timeout_ns) { tp_timeout(what, target); }
    }
}

// Advances this CTA's sequence; all CTAs of one collective read the same value.
__device__ __forceinline__ std::uint64_t tp_begin(const TensorParallelDeviceView& tp) {
    __shared__ std::uint64_t sequence;
    if (threadIdx.x == 0) {
        const std::uint64_t next = tp.counters[blockIdx.x] + 1;
        tp.counters[blockIdx.x]  = next;
        sequence                 = next;
    }
    __syncthreads();
    return sequence;
}

__device__ __forceinline__ void tp_exchange(const TensorParallelDeviceView& tp,
                                            std::uint64_t sequence) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        tp_store_release(&tp.self_mailbox->published[blockIdx.x], sequence);
        tp_wait_at_least(&tp.peer_mailbox->published[blockIdx.x], sequence, tp.timeout_ns,
                         "payload");
    }
    __syncthreads();
}

template <class T>
__device__ __forceinline__ T* tp_self_slot(const TensorParallelDeviceView& tp,
                                           std::uint64_t sequence) {
    return reinterpret_cast<T*>(tp.self_staging + (sequence % kTensorParallelSlots) * tp.slot_bytes);
}

template <class T>
__device__ __forceinline__ const T* tp_peer_slot(const TensorParallelDeviceView& tp,
                                                 std::uint64_t sequence) {
    return reinterpret_cast<const T*>(tp.peer_staging +
                                      (sequence % kTensorParallelSlots) * tp.slot_bytes);
}

__device__ __forceinline__ std::uint64_t tp_line(std::uint32_t payload, std::uint64_t sequence) {
    return (sequence << 32) | payload;
}

// Polls one packed line until it carries `sequence`; returns its payload word.
__device__ __forceinline__ std::uint32_t tp_read_line(const std::uint64_t* line,
                                                      std::uint64_t sequence,
                                                      std::uint64_t timeout_ns) {
    const std::uint32_t expected = static_cast<std::uint32_t>(sequence);
    std::uint64_t value          = tp_load_volatile(line);
    if (static_cast<std::uint32_t>(value >> 32) == expected) {
        return static_cast<std::uint32_t>(value);
    }
    const std::uint64_t start = tp_global_timer();
    for (;;) {
        value = tp_load_volatile(line);
        if (static_cast<std::uint32_t>(value >> 32) == expected) {
            return static_cast<std::uint32_t>(value);
        }
        if (tp_global_timer() - start > timeout_ns) { tp_timeout("packed payload", sequence); }
    }
}

__device__ __forceinline__ std::uint32_t tp_sum3(std::uint32_t residual, std::uint32_t first,
                                                 std::uint32_t second) {
    const __nv_bfloat162 r   = *reinterpret_cast<const __nv_bfloat162*>(&residual);
    const __nv_bfloat162 a   = *reinterpret_cast<const __nv_bfloat162*>(&first);
    const __nv_bfloat162 b   = *reinterpret_cast<const __nv_bfloat162*>(&second);
    const float lo           = (__low2float(r) + __low2float(a)) + __low2float(b);
    const float hi           = (__high2float(r) + __high2float(a)) + __high2float(b);
    const __nv_bfloat162 out = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const std::uint32_t*>(&out);
}

// Packed residual all-reduce over `words` 4-byte payload words (BF16 pairs).
__global__ __launch_bounds__(kTensorParallelThreads) void tp_residual_allreduce_packed_kernel(
    const std::uint32_t* partial, std::uint32_t* residual, std::int64_t words,
    TensorParallelDeviceView tp) {
    const std::uint64_t sequence = tp_begin(tp);
    const std::size_t slot       = (sequence % kTensorParallelSlots) * kTensorParallelPackedSlotBytes;
    auto* self                   = reinterpret_cast<std::uint64_t*>(tp.self_packed + slot);
    const auto* peer             = reinterpret_cast<const std::uint64_t*>(tp.peer_packed + slot);
    const std::int64_t stride    = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t first     = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (std::int64_t i = first; i < words; i += stride) {
        tp_store_volatile(self + i, tp_line(partial[i], sequence));
    }
    for (std::int64_t i = first; i < words; i += stride) {
        const std::uint32_t own   = partial[i];
        const std::uint32_t other = tp_read_line(peer + i, sequence, tp.timeout_ns);
        residual[i] = tp.rank == 0 ? tp_sum3(residual[i], own, other)
                                   : tp_sum3(residual[i], other, own);
    }
}

// Staged residual all-reduce; CTA b owns pack range [b * packs_per_block, ...).
__global__ __launch_bounds__(kTensorParallelThreads) void tp_residual_allreduce_kernel(
    const uint4* partial, uint4* residual, std::int64_t packs, std::int64_t packs_per_block,
    TensorParallelDeviceView tp) {
    const std::uint64_t sequence = tp_begin(tp);
    uint4* self_slot             = tp_self_slot<uint4>(tp, sequence);
    const uint4* peer_slot       = tp_peer_slot<uint4>(tp, sequence);
    const std::int64_t begin     = static_cast<std::int64_t>(blockIdx.x) * packs_per_block;
    const std::int64_t end       = begin + packs_per_block < packs ? begin + packs_per_block : packs;
    for (std::int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
        __stwt(self_slot + i, partial[i]);
    }
    tp_exchange(tp, sequence);
    for (std::int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const uint4 own    = partial[i];
        const uint4 other  = __ldcv(peer_slot + i);
        const uint4 first  = tp.rank == 0 ? own : other;
        const uint4 second = tp.rank == 0 ? other : own;
        uint4 value        = residual[i];
        value.x            = tp_sum3(value.x, first.x, second.x);
        value.y            = tp_sum3(value.y, first.y, second.y);
        value.z            = tp_sum3(value.z, first.z, second.z);
        value.w            = tp_sum3(value.w, first.w, second.w);
        residual[i]        = value;
    }
}

// `row_packs` = local rows / 8. Local pack p is column p / row_packs, row pack p % row_packs; it
// lands at column * 2 * row_packs + owner * row_packs + row pack of the destination.
__device__ __forceinline__ std::int64_t tp_gather_target(std::int64_t pack, std::int64_t row_packs,
                                                         std::int64_t owner) {
    return (pack / row_packs) * 2 * row_packs + owner * row_packs + pack % row_packs;
}

__global__ __launch_bounds__(kTensorParallelThreads) void tp_allgather_rows_kernel(
    const uint4* local, uint4* destination, std::int64_t packs, std::int64_t row_packs,
    std::int64_t packs_per_block, TensorParallelDeviceView tp) {
    const std::uint64_t sequence = tp_begin(tp);
    uint4* self_slot             = tp_self_slot<uint4>(tp, sequence);
    const uint4* peer_slot       = tp_peer_slot<uint4>(tp, sequence);
    const std::int64_t begin     = static_cast<std::int64_t>(blockIdx.x) * packs_per_block;
    const std::int64_t end       = begin + packs_per_block < packs ? begin + packs_per_block : packs;
    for (std::int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const uint4 value = local[i];
        __stwt(self_slot + i, value);
        destination[tp_gather_target(i, row_packs, tp.rank)] = value;
    }
    tp_exchange(tp, sequence);
    for (std::int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
        destination[tp_gather_target(i, row_packs, 1 - tp.rank)] = __ldcv(peer_slot + i);
    }
}

} // namespace ninfer::ops
