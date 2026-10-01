#pragma once

// Implements: include/ninfer/ops/tensor_parallel.h
// One kernel per collective. CTA b of rank r owns a contiguous pack range, stages it in rank r's
// mapped host slot, publishes its sequence number, waits for peer CTA b, then combines. Thread 0 of
// each CTA performs every mailbox access; the CTA barrier carries the ordering to the other
// threads. Host-memory payload stores are write-through and peer payload loads bypass the caches,
// because a slot address is reused every kTensorParallelSlots collectives.

#include "core/tensor_parallel.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>

namespace ninfer::ops {

inline constexpr int kTensorParallelThreads = 256;

__device__ __forceinline__ std::uint64_t tp_load_acquire(const std::uint64_t* address) {
    std::uint64_t value;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ void tp_store_release(std::uint64_t* address, std::uint64_t value) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(address), "l"(value) : "memory");
}

__device__ __forceinline__ std::uint64_t tp_global_timer() {
    std::uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__device__ __forceinline__ void tp_wait_at_least(const std::uint64_t* address, std::uint64_t target,
                                                 std::uint64_t timeout_ns, const char* what) {
    if (tp_load_acquire(address) >= target) { return; }
    const std::uint64_t start = tp_global_timer();
    while (tp_load_acquire(address) < target) {
        if (tp_global_timer() - start > timeout_ns) {
            printf("ninfer tensor-parallel: timed out waiting for peer %s (CTA %d, target %llu)\n",
                   what, static_cast<int>(blockIdx.x), static_cast<unsigned long long>(target));
            __trap();
        }
    }
}

// Advances CTA `blockIdx.x`'s sequence, waits until the peer released this rank's slot, and
// returns the sequence through shared memory to the whole CTA.
__device__ __forceinline__ std::uint64_t tp_begin(const TensorParallelDeviceView& tp) {
    __shared__ std::uint64_t sequence;
    if (threadIdx.x == 0) {
        const std::uint64_t next = tp.counters[blockIdx.x] + 1;
        tp.counters[blockIdx.x]  = next;
        if (next > kTensorParallelSlots) {
            tp_wait_at_least(&tp.peer_mailbox->consumed[blockIdx.x], next - kTensorParallelSlots,
                             tp.timeout_ns, "slot release");
        }
        sequence = next;
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

__device__ __forceinline__ void tp_finish(const TensorParallelDeviceView& tp,
                                          std::uint64_t sequence) {
    __syncthreads();
    if (threadIdx.x == 0) { tp_store_release(&tp.self_mailbox->consumed[blockIdx.x], sequence); }
}

__device__ __forceinline__ uint4* tp_self_slot(const TensorParallelDeviceView& tp,
                                               std::uint64_t sequence) {
    return reinterpret_cast<uint4*>(tp.self_staging +
                                    (sequence % kTensorParallelSlots) * tp.slot_bytes);
}

__device__ __forceinline__ const uint4* tp_peer_slot(const TensorParallelDeviceView& tp,
                                                     std::uint64_t sequence) {
    return reinterpret_cast<const uint4*>(tp.peer_staging +
                                          (sequence % kTensorParallelSlots) * tp.slot_bytes);
}

__device__ __forceinline__ std::uint32_t tp_sum3(std::uint32_t residual, std::uint32_t first,
                                                 std::uint32_t second) {
    const __nv_bfloat162 r = *reinterpret_cast<const __nv_bfloat162*>(&residual);
    const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&first);
    const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&second);
    const float lo         = (__low2float(r) + __low2float(a)) + __low2float(b);
    const float hi         = (__high2float(r) + __high2float(a)) + __high2float(b);
    const __nv_bfloat162 out = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const std::uint32_t*>(&out);
}

__global__ __launch_bounds__(kTensorParallelThreads) void tp_residual_allreduce_kernel(
    const uint4* partial, uint4* residual, std::int64_t packs, std::int64_t packs_per_block,
    TensorParallelDeviceView tp) {
    const std::uint64_t sequence = tp_begin(tp);
    uint4* self_slot             = tp_self_slot(tp, sequence);
    const uint4* peer_slot       = tp_peer_slot(tp, sequence);
    const std::int64_t begin     = static_cast<std::int64_t>(blockIdx.x) * packs_per_block;
    const std::int64_t end       = begin + packs_per_block < packs ? begin + packs_per_block : packs;
    for (std::int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
        __stwt(self_slot + i, partial[i]);
    }
    tp_exchange(tp, sequence);
    for (std::int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const uint4 own   = partial[i];
        const uint4 other = __ldcv(peer_slot + i);
        const uint4 first = tp.rank == 0 ? own : other;
        const uint4 second = tp.rank == 0 ? other : own;
        uint4 value        = residual[i];
        value.x            = tp_sum3(value.x, first.x, second.x);
        value.y            = tp_sum3(value.y, first.y, second.y);
        value.z            = tp_sum3(value.z, first.z, second.z);
        value.w            = tp_sum3(value.w, first.w, second.w);
        residual[i]        = value;
    }
    tp_finish(tp, sequence);
}

// `row_packs` = local rows / 8. Local pack p is column p / row_packs, row pack p % row_packs; it
// lands at column * 2 * row_packs + owner * row_packs + row pack of the destination.
__global__ __launch_bounds__(kTensorParallelThreads) void tp_allgather_rows_kernel(
    const uint4* local, uint4* destination, std::int64_t packs, std::int64_t row_packs,
    std::int64_t packs_per_block, TensorParallelDeviceView tp) {
    const std::uint64_t sequence = tp_begin(tp);
    uint4* self_slot             = tp_self_slot(tp, sequence);
    const uint4* peer_slot       = tp_peer_slot(tp, sequence);
    const std::int64_t begin     = static_cast<std::int64_t>(blockIdx.x) * packs_per_block;
    const std::int64_t end       = begin + packs_per_block < packs ? begin + packs_per_block : packs;
    const auto target            = [&](std::int64_t pack, std::int64_t owner) {
        return (pack / row_packs) * 2 * row_packs + owner * row_packs + pack % row_packs;
    };
    for (std::int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const uint4 value = local[i];
        __stwt(self_slot + i, value);
        destination[target(i, tp.rank)] = value;
    }
    tp_exchange(tp, sequence);
    for (std::int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
        destination[target(i, 1 - tp.rank)] = __ldcv(peer_slot + i);
    }
    tp_finish(tp, sequence);
}

} // namespace ninfer::ops
