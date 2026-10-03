#pragma once

#include <cuda_runtime.h>

#include <array>
#include <atomic>

namespace ninfer::ops::detail {

// RTX 5090 SM count; the fallback when the current device cannot be queried.
inline constexpr int kCausalAttentionSmCount = 170;
inline constexpr int kCausalAttentionDevices = 64;

// SMs of the calling thread's current device. Wave budgets remain owned by each dtype plan; split
// counts sized for a larger part leave a smaller one with many partial waves and merge traffic.
inline int causal_attention_sm_count() {
    static std::array<std::atomic<int>, kCausalAttentionDevices> counts{};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= kCausalAttentionDevices) {
        return kCausalAttentionSmCount;
    }
    auto& slot = counts[static_cast<std::size_t>(device)];
    int count  = slot.load(std::memory_order_relaxed);
    if (count == 0) {
        if (cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
            count <= 0) {
            return kCausalAttentionSmCount;
        }
        slot.store(count, std::memory_order_relaxed);
    }
    return count;
}

// Capture reserves partials for the largest row window. Producer and merge use
// the same row window; a wider capture never changes a row's work partition.
struct CausalKvPartition {
    static constexpr int kMaxSplits = 256;
    int capacity                    = 1;
    int target                      = 1;
    int key_shift                   = 6; // log2 of the minimum KV keys per split
    // log2 of the producer's key tile. Nonzero balances a row's split count down to the fewest
    // splits that keep the same largest number of key tiles per split; rows of at least
    // balance_limit keys keep the plain count.
    int balance_shift = 0;
    int balance_limit = 0;
    // SMs of the device the partition was planned for: launch-time split choices made on the
    // device (mxfp8_tiled_active_splits) must use the host plan's count.
    int sms = kCausalAttentionSmCount;
    // Query rows per CTA of a tiled route that planned this partition (0: not a tiled plan).
    int query_rows = 0;

    // The plain count: the live count of any row window up to `visible` is at most this, so
    // capture sizes partials and grids with it.
    __host__ __device__ int bound(int visible) const {
        const int count = (visible + (1 << key_shift) - 1) >> key_shift;
        return count < target ? count : target;
    }

    __host__ __device__ int active(int visible) const {
        const int count = bound(visible);
        if (balance_shift == 0 || count <= 1 || visible >= balance_limit) return count;
        const int tiles     = (visible + (1 << balance_shift) - 1) >> balance_shift;
        const int per_split = (tiles + count - 1) / count;
        return (tiles + per_split - 1) / per_split;
    }
};

// A row partitions KV as if all of its physical columns were live, so masking trailing columns
// never moves the split boundaries of the live ones. Live positions are sequential from the
// first, so a dense row's window is its last position plus one. The envelope bounds the window
// only where the physical width reaches past the visible capacity.
__host__ __device__ inline int causal_row_window(int first_position, int width,
                                                 int visible_capacity) {
    const int window = first_position + width;
    return window < visible_capacity ? window : visible_capacity;
}

} // namespace ninfer::ops::detail
