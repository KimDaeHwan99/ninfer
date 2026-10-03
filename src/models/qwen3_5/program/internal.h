#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/program.h"
#include "ninfer/ops/softmax_attention.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::models::qwen3_5 {

inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 5;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 15;
// Abort salvage publishes the live state only when it covers enough committed work that the
// saved rebuild outweighs the checkpoint's retention cost.
inline constexpr std::uint32_t kSalvageMinFrontier = 1024;

// Prompt-attention settings of every prefill chunk (CausalAttentionExecutionEnvelope): fast selects
// the storage's fast prompt kernel (INT8 or NVFP4 KV); pv8 runs the prompt kernel's PV on 8-bit
// Tensor Cores (the fast INT8 or NVFP4 kernel, or K8V4's prompt kernel, which ignores fast); and
// split_workspace_bytes bounds the FP32 partials of a launch's key splits. Workspace planning and
// every chunk use the same settings.
struct PromptAttention {
    bool fast                         = false;
    bool pv8                          = false;
    std::size_t split_workspace_bytes = ops::kCausalPromptSplitWorkspaceDefaultBytes;

    [[nodiscard]] ops::CausalAttentionExecutionEnvelope
    envelope(std::uint32_t min_visible_keys, std::uint32_t max_visible_keys) const noexcept {
        return {.min_visible_keys             = min_visible_keys,
                .max_visible_keys             = max_visible_keys,
                .fast_prompt_kernel           = fast,
                .fast_prompt_pv8              = pv8,
                .prompt_split_workspace_bytes = split_workspace_bytes};
    }
};

// Rows of 17..64 verification columns keep 16-bit activations in FP8 residual projections at any
// batch size; first/last bound the per-row width, not the round's aggregate columns.
[[nodiscard]] inline bool wide_residual_verification(TextPhase phase, std::int32_t first,
                                                     std::int32_t last) {
    return phase == TextPhase::Verify && first > 16 && first <= last && last <= 64;
}

} // namespace ninfer::models::qwen3_5

namespace ninfer::models::qwen3_5::detail {
using ContractAccess = RuntimeContractAccess;

[[nodiscard]] inline std::uint32_t backend_frontier_at(SpeculativeBackend backend,
                                                       std::uint32_t main_frontier) noexcept {
    if (backend == SpeculativeBackend::Mtp) { return main_frontier == 0 ? 0U : main_frontier - 1U; }
    return backend == SpeculativeBackend::DFlash ? main_frontier : 0U;
}

} // namespace ninfer::models::qwen3_5::detail
