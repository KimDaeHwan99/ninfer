#pragma once

// The NVFP4 [8192,5120] tensor-parallel shard of gdn_input_proj: one rank's head-local half of the
// [16384,5120] parent, rows qkv|z = 5120|3072 (query 1024, key 1024, value 3072, z 3072).
//
// The shard runs the parent's own NVFP4 projection sources, compiled a second time by the
// translation units of this directory with the section output and the entry points renamed
// (nvfp4_gdn_input_shard_names.h); both share K, so routes, schedules, cutoffs and the workspace
// capacity are the parent's by construction. The fused snapshot routes register only the parent:
// the wrapper composes the shard's projection with the generic projected convolution, as for the
// FP8 shard. Technique from ValerioDolci/ninfer-tp2.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear/nvfp4/nvfp4_operands.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kNvfp4GdnInputShardRows = 8192;

[[nodiscard]] std::size_t nvfp4_gdn_input_shard_workspace_capacity_bytes(LinearPolicy policy,
                                                                         std::int32_t min_tokens,
                                                                         std::int32_t max_tokens);
void nvfp4_gdn_input_shard_a16_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                      Tensor& z, cudaStream_t stream);
void nvfp4_gdn_input_shard_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                         Tensor& z, cudaStream_t stream);
void nvfp4_gdn_input_shard_small_t_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                          Tensor& z, cudaStream_t stream);
void nvfp4_gdn_input_shard_a4_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                     Tensor& z, Nvfp4A4Workspace workspace, cudaStream_t stream);
// In the non-RDC archive, like launch_nvfp4_a4_tma_gdn().
void launch_nvfp4_a4_tma_gdn_shard(const Nvfp4A4Operands& p, __nv_bfloat16* qkv,
                                   __nv_bfloat16* z, cudaStream_t stream);
// nvfp4_gdn_input_dispatch() at the shard shape.
void nvfp4_gdn_input_shard_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    LinearPolicy policy, WorkspaceArena* workspace,
                                    cudaStream_t stream);

} // namespace ninfer::ops::detail
