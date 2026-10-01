#pragma once

// ninfer::ops - two-rank tensor-parallel collectives over core's TensorParallelLink.
//
// Both ranks call the same collective with same-shaped operands, in the same order, on their own
// device stream. Each call is one kernel: it publishes this rank's contribution through mapped host
// memory, waits for the peer's matching contribution, and combines. The calls hold no host
// synchronization and are CUDA Graph capturable; the link's device-resident sequence numbers make
// eager execution and graph replays one numbering.

#include "core/tensor.h"
#include "core/tensor_parallel.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Row-parallel residual update:
 *
 *   ideal[i] = residual[i] + partial_rank0[i] + partial_rank1[i]
 *
 * written to `residual` on both ranks. `partial` is this rank's contiguous BF16 contribution and
 * `residual` the rank's replica of the BF16 residual stream; both have the same shape on both
 * ranks and must not overlap. Every rank evaluates the same FP32 expression in rank order and
 * rounds once, so the two residual replicas stay bit-identical. The payload must fit the link's
 * slot (`tp.slot_bytes`).
 */
void tp_residual_allreduce(const Tensor& partial, Tensor& residual,
                           const TensorParallelDeviceView& tp, cudaStream_t stream);

/**
 * Exact row gather of an equal two-way split along ne[0]:
 *
 *   destination[rank * R + i, c] = local_rank[i, c]   for i < R = local.ne[0]
 *
 * on both ranks. `local` is [R, C] and `destination` [2R, C], both contiguous BF16 and
 * non-overlapping. R must be a multiple of 8. Used for vocabulary-split output heads.
 */
void tp_allgather_rows(const Tensor& local, Tensor& destination, const TensorParallelDeviceView& tp,
                       cudaStream_t stream);

} // namespace ninfer::ops
