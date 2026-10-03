#pragma once
#include "models/qwen3_5/program/program.h"
#include "models/qwen3_5/program/round_buffers.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

[[nodiscard]] std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity);
[[nodiscard]] std::vector<GraphExecutionProfile>
mtp_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window, std::uint32_t neural_drafts);
[[nodiscard]] std::vector<GraphExecutionProfile> dflash_graph_profiles(SpeculativeBackend backend,
                                                                       std::uint32_t capacity,
                                                                       std::uint32_t draft_window);

// One MTP decode graph family: the round's verify width and the MTP head's proposal depth.
struct MtpGraphFamily {
    std::uint32_t verify_drafts = 0;
    std::uint32_t ar_depth      = 0;
    bool ngram                  = false;
};

// The MTP families an engine captures, planned once for planning, capture and decode. Neural
// rounds verify at the neural window; with copy drafting at another window, copy rounds verify at
// that window on the same frame (MtpDecodeState::narrowed), so a wide copy window costs only the
// rounds that carry a copy. The head always proposes --draft-tokens.
[[nodiscard]] inline std::vector<MtpGraphFamily> mtp_graph_families(std::uint32_t neural_window,
                                                                    std::uint32_t ngram_window) {
    const std::uint32_t ar_depth = std::min(neural_window, kMtpDecodeMaximumDrafts);
    std::vector<MtpGraphFamily> families{{neural_window, ar_depth, false}};
    if (ngram_window != 0 && ngram_window != neural_window) {
        families.push_back({ngram_window, ar_depth, true});
    }
    return families;
}

} // namespace ninfer::models::qwen3_5::detail
