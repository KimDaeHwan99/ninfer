#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer {

using TokenId = std::int32_t;

inline constexpr std::uint32_t kMaximumConcurrency               = 8;
inline constexpr std::size_t kMaximumContextCacheSessionKeyBytes = 256;
inline constexpr std::size_t kMaximumExplicitPromptCacheMarkers  = 4;
// Explicit markers plus the engine's automatic tool/leading-instruction/full-prompt candidates;
// one request's shared-prefix opportunities never exceed this (frontend.cpp opportunities.reserve).
inline constexpr std::size_t kMaximumPreparedPromptCacheCandidatesPerRequest = 7;
// Aggregate encoded image/video payload retained by one prompt, independent of item count.
inline constexpr std::size_t kMaximumPromptMediaBytes    = 256ULL << 20;
inline constexpr std::size_t kDefaultMediaCacheBytes     = 1ULL << 30;
inline constexpr std::size_t kDefaultMediaLiveBytes      = 2ULL << 30;
inline constexpr std::uint32_t kDefaultHostStateSlots    = 8;
inline constexpr std::size_t kDefaultHostKvCapacityBytes = 8ULL << 30;
// Pinned Host tier of the hybrid prefix cache when --host-cache-mib is not given.
inline constexpr std::size_t kDefaultHybridHostCacheBytes = 8ULL << 30;
// Prompt-attention split workspace when --prefill-split-workspace-mib is not given.
inline constexpr std::uint32_t kDefaultPrefillSplitWorkspaceMiB = 256;
inline constexpr std::uint32_t kMaximumPrefillSplitWorkspaceMiB = 16384;

enum class KvCacheStorage : std::uint8_t {
    BFloat16,
    Int8Group64,
    Fp8E4M3Row256,
    Nvfp4Group16,
    Fp8KeyNvfp4Value,
    // Rotated 2-bit eight-value vector codes for K and V plus an exact INT8 recent-key window.
    Vq2,
    // Rotated 4-bit Lloyd-Max K codes, 2-bit vector V codes and the same exact recent-key window.
    Q4KeyVq2Value,
};

// How prompt attention runs P*V where its kernel has an 8-bit form (the fast INT8 and NVFP4
// prompt kernels, K8V4's tiled kernel and the VQ2 and K4V2 prompt kernel). Auto keeps FP16 P*V
// for INT8 and K4V2 KV, whose 8-bit forms cost measurable KL divergence against FP16's at 64K
// context, and uses the 8-bit form for NVFP4, K8V4 and VQ2 KV, where it is numerically equivalent
// or nearly so (see the README).
enum class PrefillPv8 : std::uint8_t {
    Auto,
    On,
    Off,
};

[[nodiscard]] constexpr const char* prefill_pv8_name(PrefillPv8 policy) noexcept {
    switch (policy) {
    case PrefillPv8::Auto: return "auto";
    case PrefillPv8::On: return "on";
    case PrefillPv8::Off: return "off";
    }
    return "auto";
}

enum class EnginePurpose : std::uint8_t {
    Generation,
    CausalScoring,
};

enum class KvCapacityMode : std::uint8_t {
    Explicit,
    Automatic,
};

inline constexpr std::size_t kDefaultKvCapacityHeadroomBytes = 1024ULL * 1024ULL * 1024ULL;

struct KvCapacityPolicy {
    KvCapacityMode mode                  = KvCapacityMode::Explicit;
    std::uint32_t explicit_tokens        = 2048;
    std::size_t automatic_headroom_bytes = 0;

    [[nodiscard]] static constexpr KvCapacityPolicy
    explicit_capacity(std::uint32_t tokens) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Explicit, tokens, 0};
    }

    [[nodiscard]] static constexpr KvCapacityPolicy
    automatic(std::size_t headroom_bytes = kDefaultKvCapacityHeadroomBytes) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Automatic, 0, headroom_bytes};
    }
};

enum class ProposalHead : std::uint8_t {
    Full,
    Optimized,
};

enum class SpeculativeBackend : std::uint8_t {
    None,
    Mtp,
    DFlash,
    DFlash2,
};

// Automatic DFlash2 tree verification considers, besides the chain, trees of these many columns
// (anchor included) for all-neural rounds of at most kDraftTreeAutoMaxBatch rows.
inline constexpr std::uint32_t kDraftTreeAutoMaxBatch = 4;

[[nodiscard]] constexpr std::array<std::uint32_t, 2>
draft_tree_auto_widths(std::uint32_t draft_tokens) noexcept {
    return {draft_tokens + 5U, draft_tokens + 9U};
}

struct SpeculativeOptions {
    SpeculativeBackend backend = SpeculativeBackend::None;
    // Startup-fixed K: MTP 1..5; DFlash and DFlash2 1..15 (query width K+1).
    std::uint32_t draft_tokens = 0;
    ProposalHead proposal_head = ProposalHead::Full;
    // Optional target-verified copy proposals with MTP, DFlash or DFlash2, C1 only.
    // Zero disables the proposer; enabled draft width is 1..63 and minimum match 4..64.
    std::uint32_t ngram_draft_tokens = 0;
    std::uint32_t ngram_min_match    = 12;
    // DFlash2 tree verification by batch size: entry c-1 is the column count (anchor included,
    // draft_tokens+2..32) a neural round of c rows verifies per row as a draft tree built from the
    // drafter's lattice; zero keeps that batch size on chain verification. All zero disables trees.
    std::array<std::uint32_t, kMaximumConcurrency> draft_tree_nodes{};
    // Automatic DFlash2 tree verification (draft_tree_nodes all zero): each all-neural round
    // verifies a chain or one of draft_tree_auto_widths, whichever measured round time and
    // same-text acceptance favour at its batch size and context length. A target whose GDN input
    // projections cannot verify trees resolves this to false (chain verification).
    bool draft_tree_auto = false;
    // Most root-to-leaf paths a tree row may hold, 2..8.
    std::uint32_t draft_tree_paths = 8;
    // CPU-only retention, separate from KV. Zero keeps request-local drafting.
    std::size_t ngram_archive_bytes = 0;
    std::size_t ngram_session_bytes = 128ULL << 20;
};

enum class StartupPhase : std::uint8_t {
    EngineStartup,
    CudaInitialize,
    ArtifactInspect,
    TargetPlan,
    WeightsMaterialize,
    WeightsStagingPin,
    TargetFinalize,
    FrontendInitialize,
    ProgramInitialize,
    HostStatePin,
    HostKvPin,
    CudaGraphPrepare,
    PrefixCacheLoad,
    EngineFinalize,
};

enum class StartupStatus : std::uint8_t {
    Begin,
    Progress,
    Complete,
    Failed,
};

enum class StartupProgressUnit : std::uint8_t {
    None,
    Bytes,
};

struct StartupEvent {
    StartupPhase phase                = StartupPhase::EngineStartup;
    StartupStatus status              = StartupStatus::Begin;
    StartupProgressUnit progress_unit = StartupProgressUnit::None;
    std::uint64_t current             = 0;
    std::uint64_t total               = 0;
    std::uint64_t elapsed_ns          = 0;
};

struct StartupObserver {
    // Startup diagnostics never participate in Engine control flow. Callback exceptions are
    // ignored by the publishing boundary so a logging failure cannot invalidate model startup.
    std::function<void(const StartupEvent& event)> callback;
};

enum class DiagnosticLevel : std::uint8_t {
    Debug,
    Info,
    Warning,
    Error,
};

// One runtime diagnostic from the Engine worker: Device KV lease growth, recovery from out of
// memory or a failed request, and prefix-cache persistence at shutdown.
struct Diagnostic {
    DiagnosticLevel level = DiagnosticLevel::Info;
    std::string message;
};

struct DiagnosticObserver {
    // Receives every diagnostic, from the worker thread or the destructor; the product decides
    // which levels to show. Without a callback, Info and above go to stderr. Callback exceptions
    // are ignored so a logging failure cannot disturb execution.
    std::function<void(const Diagnostic& diagnostic)> callback;
};

// Prefix-cache implementation selected at Engine construction. Legacy is the owner/checkpoint
// ResourceManager (docs/maintainer/resource-scheduling-and-context-cache.md). Hybrid is the
// content-addressed block tree with sparse state snapshots
// (docs/maintainer/hybrid-prefix-cache-spec.md).
// Prefix-cache design. ninfer-serve selects Hybrid unless --use-original-prefix-caching selects
// Legacy; the Engine option itself defaults to Legacy.
enum class ContextCacheMode : std::uint8_t {
    Legacy,
    Hybrid,
};

// Abandons the Host tier save of a stopping Engine (HybridPrefixCacheOptions::persistent_file)
// from any thread, also while the Engine is being destroyed. Copies share one state: the product
// keeps a copy from the options it passes, beyond the Engine's lifetime.
class PrefixCacheSaveControl {
public:
    enum class Abandon : std::uint8_t {
        // Nothing replaced the previous file; a save in progress deleted its unfinished file.
        Unsaved,
        // The save had already completed.
        Saved,
        // The save did not stop within the timeout; its unfinished file may remain.
        StillWriting,
    };

    PrefixCacheSaveControl();

    // Stops a save in progress, or one not begun yet, from replacing the previous file. Waits at
    // most `timeout` for a save in progress to stop and delete its unfinished file.
    Abandon abandon(std::chrono::milliseconds timeout) const noexcept;

    // The Engine's side. A save writes only after begin() returns true, polls abandoned() while
    // it writes, and then calls end() with whether it replaced the file.
    [[nodiscard]] bool begin() const noexcept;
    [[nodiscard]] bool abandoned() const noexcept;
    void end(bool saved) const noexcept;

private:
    struct State;
    std::shared_ptr<State> state_;
};

// Hybrid-mode tuning. Every field is optional: Engine construction derives the unset ones from
// max_concurrency, prefill_chunk and whether a Host tier exists (host_cache_budget_bytes, default
// kDefaultHybridHostCacheBytes, 0 disables it), and Engine::options() reports the effective
// values. Legacy-only fields of ContextCacheOptions are rejected in Hybrid mode.
struct HybridPrefixCacheOptions {
    // Device StateImage slots holding inactive snapshots (and tap/endpoint staging). Total Device
    // StateImage capacity is max_concurrency + device_snapshot_slots. Default: one per request
    // lane plus one staging slot with a Host tier, plus two without one (Device slots are then
    // the only snapshot storage).
    std::optional<std::uint32_t> device_snapshot_slots;
    // New prefill state snapshots one request may create. Default 8 with a Host tier, 2 without.
    std::optional<std::uint32_t> max_new_taps;
    // Geometric ladder base G: ladder taps target prompt_tokens - G * 2^k. Default
    // max(4096, 2 * prefill_chunk): ladder taps land on prefill chunk boundaries.
    std::optional<std::uint32_t> tap_ladder_tokens;
    // Ladder taps closer than this to another snapshot on the same path are skipped. Default
    // max(1024, prefill_chunk).
    std::optional<std::uint32_t> tap_min_gap_tokens;
    // Opt-in persistence of the Host tier: saved to this file when the Engine shuts down cleanly
    // and restored from it at startup, but only when it was written for the same artifact, KV and
    // state formats and `persistent_identity` (the product binary's build). Empty disables it.
    std::filesystem::path persistent_file;
    std::string persistent_identity;
    // Lets the product abandon the save at shutdown (ninfer-serve: Ctrl+C during the stop).
    PrefixCacheSaveControl persistent_save;
};

struct ContextCacheOptions {
    // Engine resolves every optional once at construction. With C=max_concurrency, the enabled
    // defaults are H=C, R=8, Host KV=8 GiB, P=2C, S=max(C,7) and L=4;
    // Engine::options() returns those effective values.
    bool enabled = true;
    ContextCacheMode mode = ContextCacheMode::Legacy;
    HybridPrefixCacheOptions hybrid;
    // Extra Device checkpoint StateImage slots H. Total Device StateImage capacity is C + H.
    std::optional<std::uint32_t> device_state_slots;
    // Host StateImages and Host KV bytes are independently configured pinned-memory capacities.
    std::uint32_t host_state_slots     = kDefaultHostStateSlots;
    std::size_t host_kv_capacity_bytes = kDefaultHostKvCapacityBytes;
    // Single host RAM ceiling for the whole retention tier. In Hybrid mode it is the pinned Host
    // slab pool that KV blocks and state snapshots share (default kDefaultHybridHostCacheBytes, 0
    // disables the Host tier). In Legacy mode, when engaged it is authoritative:
    // the plan sizes the Host state pool from the checkpoint inventory the capture path creates
    // (2 + anchors per private owner plus one per shared entry), spends the remaining state
    // headroom under the half-budget cap on extra long anchors per owner, gives Host KV the
    // remainder, and rejects a plan whose state footprint exceeds half the budget.
    // `host_state_slots` and `host_kv_capacity_bytes` are ignored in that mode.
    std::optional<std::size_t> host_cache_budget_bytes;
    // Bounded private/shared logical catalogs and per-continuation long-anchor count. An engaged
    // host-cache budget raises the anchor count within the state inventory it funds.
    std::optional<std::uint32_t> max_private_continuations;
    std::optional<std::uint32_t> max_shared_prefixes;
    std::optional<std::uint32_t> max_long_anchors_per_continuation;
    // Minimum token gap between the engine-automatic long anchors of one request, doubling per
    // anchor walking back from the prompt end (gap k >= spacing * 2^k). Anchors closer than that
    // cost a prefill split and a full StateImage each while covering little the endpoint and
    // rewrite checkpoints do not, so the grid is sparse near the end and reaches deep history.
    // Zero disables the spacing rule (every one of the last L message boundaries is anchored).
    std::uint32_t long_anchor_min_spacing_tokens = 1024;
};

struct ContextCostOptions {
    // Empty selects generic defaults plus any matching values compiled into the binary. A
    // nonempty runtime preset independently overrides its matching machine transfer and
    // artifact-prefill components; absent entries retain the preceding numerical layer.
    std::filesystem::path preset_path;
};

struct EngineOptions {
    std::filesystem::path artifact_path;
    std::filesystem::path chat_template_path;
    // Message the model receives when it hits its thinking budget, before the canonical
    // </think> close the frontend appends when the message lacks it. Empty preserves the
    // model's built-in end-of-thinking control suffix.
    std::string thinking_budget_message;
    EnginePurpose purpose              = EnginePurpose::Generation;
    int device                         = 0;
    // Tensor-parallel width (1 or 2). Each rank runs on `devices[rank]`; at width 1 an empty list
    // selects `device`. Width 2 splits one resident model across both devices.
    std::uint32_t tensor_parallel = 1;
    std::vector<int> devices;
    std::uint32_t max_context          = 2048; // Logical ceiling of one request or score window.
    // Startup-fixed runtime YaRN override, finite [1,4]; 1 preserves native RoPE.
    // Extends only the allowed ceiling, never max_context itself or artifact metadata.
    float rope_yarn_factor             = 1.0F;
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 16;
    std::uint32_t pending_timeout_ms   = 30000;
    std::uint32_t prefill_chunk        = 1024;
    // Opt-in: prefill units rotate round-robin over the lanes with staged prefill, starting after
    // the lane the previous unit served, and while another request is active each unit advances
    // at most 1024 prompt tokens (rounded like prefill_chunk). False serves the lowest such lane
    // first in whole chunks, so a prompt staged beside a long one may wait for the whole long
    // prefill.
    bool prefill_round_robin           = false;
    // INT8 KV prefills with the fast prompt-attention kernel and rounds prefill_chunk down to whole
    // prompt-attention waves. True selects the original INT8 prompt kernel at the requested chunk;
    // it requires the INT8 KV cache.
    bool original_int8_prefill_kernel  = false;
    // Prompt attention's P*V form where its kernel has both: the fast INT8 kernel's 8-bit form
    // rounds each probability (times its key's V group scale) to a u8 code per 64-key tile and
    // multiplies the stored V codes exactly, and the fast NVFP4 and the K8V4 prompt kernels round
    // probabilities and decoded V to E4M3; both are faster long-prompt prefill for a precision
    // change of P (and NVFP4/K8V4 V). BF16 and FP8 KV and the original INT8 and NVFP4 prompt
    // kernels have only FP16 P*V, which this setting leaves alone.
    PrefillPv8 prefill_8bit_pv         = PrefillPv8::Auto;
    // Workspace (MiB, [0,16384]) the FP32 partials of one prompt-attention launch may take when
    // it splits its keys across SMs (the INT8 and NVFP4 fast prompt kernels, FP8 and K8V4).
    // Larger budgets speed up prompt chunks of about 1-2K tokens over long cached context and take
    // device memory from the KV cache; BF16 KV and the original INT8/NVFP4 kernels ignore it.
    std::uint32_t prefill_split_workspace_mib = kDefaultPrefillSplitWorkspaceMiB;
    // NVFP4 KV prefills over more than 768 visible keys with the fast prompt-attention kernel
    // (block-scaled FP4 QK). True selects the tiled NVFP4 prompt kernel; it requires the NVFP4
    // KV cache.
    bool original_nvfp4_prefill_kernel = false;
    KvCacheStorage kv_cache            = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    std::size_t media_cache_bytes = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes  = kDefaultMediaLiveBytes;
    // Zero selects a bounded worker count from the detected host concurrency.
    std::uint32_t media_preprocess_threads = 0;
    bool enable_vision                     = false;
    // Vision offload keeps the vision tower in pinned system RAM instead of device memory and
    // streams it through borrowed evictable device staging per encode window.
    bool vision_offload = false;
    // Upper bound on merged tokens per vision item; zero leaves the compiled limit.
    std::uint32_t vision_max_merged_tokens = 32768;
    bool use_cuda_graph                    = true;
    ContextCacheOptions context_cache;
    ContextCostOptions context_cost;
    StartupObserver startup_observer;
    DiagnosticObserver diagnostic_observer;
};

enum class SamplingMode : std::uint8_t {
    Thinking,
    NonThinking,
};

// Immutable model-owned values used when a request does not override a sampling field. Seed is
// deliberately excluded: it is an execution choice rather than a model recommendation.
struct SamplingPreset {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
};

struct ModelSamplingDefaults {
    SamplingPreset thinking;
    SamplingPreset non_thinking;

    [[nodiscard]] constexpr const SamplingPreset& for_mode(SamplingMode mode) const noexcept {
        return mode == SamplingMode::Thinking ? thinking : non_thinking;
    }
};

// Public request-side overrides. std::nullopt means "use the registered model/mode default";
// explicit zero remains a real override (including temperature=0 for exact argmax).
struct SamplingOverrides {
    std::optional<float> temperature;
    std::optional<std::int32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<float> presence_penalty;
    std::optional<float> frequency_penalty;
    std::optional<std::uint64_t> seed;
};

// Complete parameters after Engine resolution. Target runtimes consume only this type.
struct ResolvedSamplingParameters {
    float temperature       = 0.0F;
    std::int32_t top_k      = 20;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
    std::uint64_t seed      = 0;
};

enum class OutputChannel : std::uint8_t {
    Content,
    Reasoning,
};

struct StopString {
    std::string text;
    OutputChannel channel  = OutputChannel::Content;
    bool include_in_output = false;
};

struct StopPolicy {
    std::vector<TokenId> token_ids;
    std::vector<StopString> strings;
    bool include_model_defaults = true;
    bool publish_stop_token     = false;
};

struct ThinkingControlOptions {
    // Positive maximum accepted model-origin tokens while the Qwen thinking phase remains open.
    // Omitted means unlimited. Injected target-control tokens consume the total output budget but
    // not this model-origin budget.
    std::optional<std::uint32_t> budget;
};

struct ExecutionOptions {
    SamplingOverrides sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
    ThinkingControlOptions thinking;
};

struct OutputOptions {
    bool raw                     = false;
    bool preserve_special_tokens = false;
    // Presentation constraint supplied by the protocol adapter. It bounds only Qwen's emitted
    // function-name grammar; it does not require the name to match a currently declared tool.
    std::uint32_t tool_name_max_length = 128;
    // Opt-in recovery of complete Qwen calls whose wrapper or suffix markup is malformed; see the
    // tool-call parser tolerant mode. Off preserves the strict all-or-nothing behavior.
    bool tolerant_tool_calls = false;
};

struct NgramSessionHints {
    // Explicit local conversation identity, at most 256 bytes. Never inferred from
    // prompt similarity or the KV session key. Empty means request-local drafting.
    std::string key;
    bool reset = false;
    std::string parent;
    std::uint64_t parent_generation = 0;
    // Retention-enabled requests mix the sampling seed with fresh request entropy:
    // prior generated proposals must not reuse the random draws that produced them.
};

struct RequestOptions {
    ExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
    NgramSessionHints ngram_session;
};

enum class MediaKind : std::uint8_t {
    Image,
    Video,
};

enum class ImageResizePolicy : std::uint8_t {
    Downsize,
    RejectOversized,
};

struct OwnedMedia {
    MediaKind kind = MediaKind::Image;
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
    ImageResizePolicy image_resize_policy = ImageResizePolicy::Downsize;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

// Model-origin structured output. Protocol adapters own any wire-level call identifier.
struct GeneratedToolCall {
    std::string name;
    std::string arguments_json;
};

// Terminal interpretation of model-origin tool-call markup. Parameter schemas guide JSON
// normalization but do not validate the call; only a structure/identity failure can return a
// complete marker region to ordinary content.
enum class ToolCallParseFallbackReason : std::uint8_t {
    None,
    MalformedStructure,
    DuplicateParameter,
    InvalidToolName,
    UndeclaredTool,
    TrailingContent,
    // Tolerant recovery discarded a trailing suffix after an otherwise complete call, or kept a
    // call whose closing tags were cut off at the region end. A structured response was still
    // produced, so this is surfaced for transparency, not as a fallback-to-text failure.
    TruncatedTail,
};

[[nodiscard]] inline constexpr const char*
tool_call_parse_fallback_reason_name(ToolCallParseFallbackReason reason) noexcept {
    switch (reason) {
    case ToolCallParseFallbackReason::None:
        return "none";
    case ToolCallParseFallbackReason::MalformedStructure:
        return "malformed_structure";
    case ToolCallParseFallbackReason::DuplicateParameter:
        return "duplicate_parameter";
    case ToolCallParseFallbackReason::InvalidToolName:
        return "invalid_tool_name";
    case ToolCallParseFallbackReason::UndeclaredTool:
        return "undeclared_tool";
    case ToolCallParseFallbackReason::TrailingContent:
        return "trailing_content";
    case ToolCallParseFallbackReason::TruncatedTail:
        return "truncated_tail";
    }
    return "malformed_structure";
}

struct ToolCallParseDiagnostics {
    bool marker_seen                              = false;
    std::uint32_t structured_call_count           = 0;
    std::uint32_t empty_arguments_omitted         = 0;
    std::uint32_t schema_mismatch_arguments       = 0;
    std::uint32_t duplicate_parameters_repaired   = 0;
    ToolCallParseFallbackReason fallback_reason   = ToolCallParseFallbackReason::None;
    // Tolerant tool-call mode turned output the strict parser rejects into structured calls.
    bool tolerant_recovered = false;

    [[nodiscard]] friend constexpr bool
    operator==(const ToolCallParseDiagnostics&, const ToolCallParseDiagnostics&) noexcept = default;
};

// Wire-independent conversation authority. Protocol adapters preserve these roles and their
// ordering; a target frontend owns any model-specific role lowering.
enum class ChatRole : std::uint8_t {
    System,
    Developer,
    User,
    Assistant,
    Tool,
};

enum class MessagePartKind : std::uint8_t {
    Text,
    Media,
};

struct MessagePart {
    MessagePartKind kind = MessagePartKind::Text;
    std::string text;
    OwnedMedia media;
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::vector<MessagePart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
};

enum class ReasoningEffort : std::uint8_t {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
};

[[nodiscard]] constexpr std::string_view reasoning_effort_name(ReasoningEffort effort) noexcept {
    switch (effort) {
    case ReasoningEffort::None:
        return "none";
    case ReasoningEffort::Minimal:
        return "minimal";
    case ReasoningEffort::Low:
        return "low";
    case ReasoningEffort::Medium:
        return "medium";
    case ReasoningEffort::High:
        return "high";
    case ReasoningEffort::XHigh:
        return "xhigh";
    case ReasoningEffort::Max:
        return "max";
    }
    return {};
}

enum class PromptContinuationMode : std::uint8_t {
    NewAssistantTurn,
    ContinueFinalAssistant,
};

struct PromptOptions {
    PromptContinuationMode continuation = PromptContinuationMode::NewAssistantTurn;
    std::optional<bool> enable_thinking;
    std::optional<ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    // JSON object of template parameters. Unset typed fields leave template defaults intact.
    std::string chat_template_kwargs_json;
    bool add_vision_id = false;
    std::vector<std::string> tool_jsons;
};

enum class CacheRetentionHint : std::uint8_t {
    Default,
    LiveSession,
    Disposable,
};

enum class PromptCacheMarkerKind : std::uint8_t {
    SharedStablePrefix,
    PrivateLongAnchor,
};

enum class SharedCandidateEvidence : std::uint8_t {
    None               = 0,
    ExplicitBoundary   = 1U << 0U,
    RequestedAutomatic = 1U << 1U,
    DefaultAutomatic   = 1U << 2U,
    EngineStructural   = 1U << 3U,
    EngineObserved     = 1U << 4U,
};

[[nodiscard]] constexpr SharedCandidateEvidence operator|(SharedCandidateEvidence left,
                                                          SharedCandidateEvidence right) noexcept {
    return static_cast<SharedCandidateEvidence>(static_cast<std::uint8_t>(left) |
                                                static_cast<std::uint8_t>(right));
}

constexpr SharedCandidateEvidence& operator|=(SharedCandidateEvidence& left,
                                              SharedCandidateEvidence right) noexcept {
    left = left | right;
    return left;
}

[[nodiscard]] constexpr bool has_shared_candidate_evidence(SharedCandidateEvidence value,
                                                           SharedCandidateEvidence evidence) {
    return (static_cast<std::uint8_t>(value) & static_cast<std::uint8_t>(evidence)) != 0;
}

enum class PromptCacheMarkerLocation : std::uint8_t {
    MessageBoundary,
    MessagePartBoundary,
    LeadingInstructionBoundary,
    ToolBoundary,
};

struct PromptCacheMarker {
    std::uint32_t after_message_count  = 0;
    PromptCacheMarkerKind kind         = PromptCacheMarkerKind::SharedStablePrefix;
    SharedCandidateEvidence evidence   = SharedCandidateEvidence::ExplicitBoundary;
    PromptCacheMarkerLocation location = PromptCacheMarkerLocation::MessageBoundary;
    // Byte count within the untrimmed leading System/Developer message.
    std::uint32_t leading_instruction_bytes = 0;
    std::uint32_t after_tool_count          = 0;
    // For MessagePartBoundary, after_message_count identifies the containing message using a
    // one-based count and this value identifies the number of serialized parts within it.
    std::uint32_t after_message_part_count = 0;

    [[nodiscard]] friend constexpr bool operator==(PromptCacheMarker,
                                                   PromptCacheMarker) noexcept = default;
};

struct ContextCacheHints {
    std::optional<std::string> session_key;
    CacheRetentionHint retention = CacheRetentionHint::Default;
    std::vector<PromptCacheMarker> markers;
    // Protocols with their own automatic/explicit write policy disable the Engine's structural
    // candidates. Exact reads from already-published shared prefixes remain enabled.
    bool allow_engine_automatic_shared_prefixes = true;
    // Advance the named session lineage when session_key is present. This does not require an
    // anonymous content-matched source to be retained.
    bool update_session_index = true;
};

struct PromptInput {
    std::vector<ChatMessage> messages;
    PromptOptions options;
    ContextCacheHints context_cache;
};

enum class RequestErrorKind : std::uint8_t {
    ContextLengthExceeded,
    ThinkingBudgetCapacityInsufficient,
    MediaBudgetExceeded,
    InvalidMedia,
    Overloaded,
    QueueTimeout,
    Cancelled,
    Unavailable,
};

class RequestError final : public std::invalid_argument {
public:
    RequestError(RequestErrorKind kind, std::string message)
        : std::invalid_argument(std::move(message)), kind_(kind) {}

    [[nodiscard]] RequestErrorKind kind() const noexcept { return kind_; }

private:
    RequestErrorKind kind_;
};

struct PromptSummary {
    bool starts_in_reasoning    = false;
    std::uint32_t prompt_tokens = 0;
    bool has_media              = false;
};

struct PromptPreparationStats {
    double seconds                       = 0.0;
    double media_preprocess_seconds      = 0.0;
    double media_preprocess_work_seconds = 0.0;
    double tokenize_seconds              = 0.0;
    std::size_t media_items              = 0;
    std::size_t media_bytes              = 0;
    std::uint64_t raw_patches            = 0;
    std::uint64_t vision_tokens          = 0;
    std::size_t patch_bytes              = 0;
    std::size_t media_cache_hits         = 0;
    std::size_t media_cache_misses       = 0;
    std::size_t media_singleflight_waits = 0;
    std::size_t built_patch_bytes        = 0;
    std::size_t reused_patch_bytes       = 0;
};

struct MediaCacheSummary {
    std::size_t capacity_bytes       = 0;
    std::size_t live_capacity_bytes  = 0;
    std::size_t retained_bytes       = 0;
    std::size_t live_bytes           = 0;
    std::size_t entries              = 0;
    std::size_t inflight             = 0;
    std::size_t queued_tasks         = 0;
    std::size_t active_tasks         = 0;
    std::uint32_t preprocess_threads = 0;
    std::uint64_t hits               = 0;
    std::uint64_t misses             = 0;
    std::uint64_t singleflight_waits = 0;
    std::uint64_t evictions          = 0;
    std::uint64_t oversize_bypasses  = 0;
};

enum class FinishReason : std::uint8_t {
    None,
    OutputLimit,
    ContextCapacity,
    StopToken,
    StopString,
    Cancelled,
};

struct OutputDelta {
    OutputChannel channel = OutputChannel::Content;
    std::string text;
};

// Exact prompt accounting selected at admission. Streaming consumers receive this once before any
// OutputDelta, after the prefix choice and materialization reservation are committed and before
// transfer/prefill execution.
struct GenerationStart {
    PromptSummary prompt;
    std::uint32_t reused_prompt_tokens = 0;
};

// Cumulative prompt frontier published only after the corresponding Program work has completed.
// The Engine owns the request-relative clock and accounting; protocol adapters choose names and
// units for the wire representation.
struct PromptProgress {
    std::uint32_t total_prompt_tokens     = 0;
    std::uint32_t reused_prompt_tokens    = 0;
    std::uint32_t processed_prompt_tokens = 0;
    std::uint64_t elapsed_ns              = 0;
};

// Cumulative timing snapshot at one stable output-commit boundary. Generated tokens count model
// and Engine-injected tokens accepted into the sequence, independently of whether the Frontend has
// enough visible bytes to publish an OutputDelta for that boundary.
struct GenerationTimingObservation {
    std::uint32_t generated_tokens      = 0;
    std::uint64_t prompt_elapsed_ns     = 0;
    std::uint64_t generation_elapsed_ns = 0;
};

class OutputSink {
public:
    virtual ~OutputSink()                                   = default;
    virtual void start(GenerationStart start)               = 0;
    virtual void progress(PromptProgress progress)          = 0;
    virtual void timing(GenerationTimingObservation timing) = 0;
    virtual void publish(OutputDelta delta)                 = 0;
};

enum class OutputConsumerMode : std::uint8_t {
    Aggregate,
    Streaming,
};

// Observation affects only request publication. It never changes model execution, output
// semantics, scheduling, or cache selection. Live observations require a Streaming consumer;
// phase timings may also be retained for an Aggregate terminal response.
struct GenerationObservationOptions {
    bool phase_timings   = false;
    bool live_timings    = false;
    bool prompt_progress = false;
};

class CancellationView {
public:
    CancellationView() = default;
    explicit CancellationView(std::function<bool()> requested);

    [[nodiscard]] bool requested() const;

private:
    std::function<bool()> requested_;
};

// Deadline and cancellation apply to all host-side prompt preparation work. Empty values mean
// unbounded preparation.
struct PreparationControl {
    std::chrono::steady_clock::time_point deadline;
    CancellationView cancellation;
};

// Request-stage wall timings retained for end-to-end latency/rate reporting. Prefill/decode are
// Program execution elapsed time and include Device completion waits; total also includes queueing
// and other request lifetime. They are not Host-work phases. GenerationEngineTiming below is the
// direct, mutually-exclusive Host observation contract.
struct GenerationTimings {
    double prepare_seconds     = 0.0;
    double first_token_seconds = 0.0;
    double vision_seconds      = 0.0;
    double prefill_seconds     = 0.0;
    double decode_seconds      = 0.0;
    // decode_seconds with each compact-batch unit's elapsed time divided among the requests it
    // served. Unlike decode_seconds, summing it over concurrent requests counts each unit once, so
    // summed decode tokens over summed shares is aggregate decode throughput.
    double decode_share_seconds = 0.0;
    // Request wall phases at committed model-state boundaries. Prompt begins when admission and
    // its exact reuse choice are published and ends at the first accepted output token. Generation
    // spans the first through last accepted output token and therefore has N-1 token intervals.
    double prompt_wall_seconds     = 0.0;
    double generation_wall_seconds = 0.0;
    double total_seconds           = 0.0;
    // Vision offload: duration and traffic of the encode window that streamed the vision tower
    // from system RAM through borrowed device memory.
    double vision_offload_window_seconds       = 0.0;
    double vision_offload_evict_seconds        = 0.0;
    double vision_offload_restore_seconds      = 0.0;
    std::uint64_t vision_offload_evicted_bytes = 0;
    std::uint64_t vision_offload_staged_bytes  = 0;
};

// Wall elapsed time directly observed in Engine-owned regions. "Exposed" values are latency
// exposure: every active request delayed by one compact-batch unit observes that unit's full
// elapsed time, so values from concurrent requests must not be summed. Device wait is reported
// separately from Host-active work.
struct GenerationEngineTiming {
    double queue_wait_seconds                   = 0.0;
    double engine_boundary_exposed_seconds      = 0.0;
    double program_submit_exposed_seconds       = 0.0;
    double program_post_exposed_seconds         = 0.0;
    double engine_commit_output_exposed_seconds = 0.0;
    double engine_maintenance_exposed_seconds   = 0.0;
    double device_wait_exposed_seconds          = 0.0;
    double decode_host_exposed_seconds          = 0.0;
    double decode_device_wait_exposed_seconds   = 0.0;
    std::uint64_t prefill_units                 = 0;
    std::uint64_t decode_rounds                 = 0;
    std::uint64_t control_units                 = 0;
};

struct SpeculativeStats {
    SpeculativeBackend backend    = SpeculativeBackend::None;
    bool enabled                  = false;
    std::uint32_t draft_window    = 0;
    std::uint64_t rounds          = 0;
    std::uint64_t drafted_tokens  = 0;
    std::uint64_t accepted_tokens = 0;
    std::uint64_t fallback_steps  = 0;
    std::vector<std::uint64_t> accepted_per_position;
    std::uint64_t ngram_rounds                  = 0;
    std::uint64_t ngram_drafted_tokens          = 0;
    std::uint64_t ngram_accepted_tokens         = 0;
    std::uint64_t ngram_archive_rounds          = 0;
    std::uint64_t ngram_archive_drafted_tokens  = 0;
    std::uint64_t ngram_archive_accepted_tokens = 0;
    // Tree-verified rounds, those whose accepted path left the main chain, and the drafts those
    // paths accepted after leaving it (what the main chain alone would not have accepted).
    std::uint64_t tree_rounds               = 0;
    std::uint64_t tree_side_rounds          = 0;
    std::uint64_t tree_side_accepted_tokens = 0;
};

struct ThinkingBudgetStats {
    std::optional<std::uint32_t> configured_budget;
    // Model-origin tokens accepted while capped thinking remained open.
    std::uint32_t model_thinking_tokens = 0;
    // Complete tokenizer-derived target-control suffix committed by Engine.
    std::uint32_t injected_tokens = 0;
    bool applied                  = false;
};

enum class PrefixReusePath : std::uint8_t {
    Root,
    PrivateEndpoint,
    PrivateTurnClosure,
    PrivateResponseReplay,
    PrivateLongAnchor,
    SharedStablePrefix,
};

// Why bounded pressure planning stopped for the materialization decision committed to one request.
enum class MaterializationStopReason : std::uint8_t {
    NoPressure,
    QueueExhausted,
    TargetBudget,
    ExpansionCapacity,
    TimeBudget,
    InsufficientExpectedGain,
    WorkBudget,
};

[[nodiscard]] inline constexpr const char*
materialization_stop_reason_name(MaterializationStopReason reason) noexcept {
    switch (reason) {
    case MaterializationStopReason::NoPressure:
        return "no_pressure";
    case MaterializationStopReason::QueueExhausted:
        return "queue_exhausted";
    case MaterializationStopReason::TargetBudget:
        return "target_budget";
    case MaterializationStopReason::ExpansionCapacity:
        return "expansion_capacity";
    case MaterializationStopReason::TimeBudget:
        return "time_budget";
    case MaterializationStopReason::InsufficientExpectedGain:
        return "insufficient_expected_gain";
    case MaterializationStopReason::WorkBudget:
        return "work_budget";
    }
    return "no_pressure";
}

enum class MaterializationSearchPhase : std::uint8_t {
    None,
    Setup,
    Construction,
    Assessment,
    Expansion,
    Refinement,
};

[[nodiscard]] inline constexpr const char*
materialization_search_phase_name(MaterializationSearchPhase phase) noexcept {
    switch (phase) {
    case MaterializationSearchPhase::None:
        return "none";
    case MaterializationSearchPhase::Setup:
        return "setup";
    case MaterializationSearchPhase::Construction:
        return "construction";
    case MaterializationSearchPhase::Assessment:
        return "assessment";
    case MaterializationSearchPhase::Expansion:
        return "expansion";
    case MaterializationSearchPhase::Refinement:
        return "refinement";
    }
    return "none";
}

struct MaterializationDiagnostics {
    std::uint64_t predicted_now_ns           = 0;
    std::uint64_t predicted_future_loss_ns   = 0;
    std::uint64_t predicted_total_ns         = 0;
    std::uint32_t targets_evaluated          = 0;
    std::uint64_t projection_work            = 0;
    std::uint64_t planning_elapsed_ns        = 0;
    std::uint64_t search_elapsed_ns          = 0;
    MaterializationStopReason stop_reason    = MaterializationStopReason::NoPressure;
    bool budget_exhausted                    = false;
    std::uint32_t selected_degradation_units = 0;
    bool selected_maximal_fallback           = false;

    std::uint64_t initial_predicted_total_ns = 0;
    std::optional<std::uint64_t> first_improvement_ns;
    std::uint32_t incumbent_improvements         = 0;
    std::uint64_t search_work                    = 0;
    std::uint64_t search_granted_ns              = 0;
    std::uint32_t search_renewals                = 0;
    bool search_discovery_used                   = false;
    std::uint64_t search_overshoot_ns            = 0;
    MaterializationSearchPhase search_stop_phase = MaterializationSearchPhase::None;
    bool search_boundary_limited                 = false;

    // Hybrid prefix cache admission. `cached_prefix_tokens` is the longest prompt prefix held as
    // cached KV blocks whether or not it was reusable: reuse also needs a state snapshot inside
    // it, so a gap to the reused token count is prefix lost to snapshot placement.
    // `restored_host_bytes` is what the admission copied back from the Host tier; the copies
    // overlap the request's first prefill pass, so their time is part of its prefill.
    std::uint32_t cached_prefix_tokens = 0;
    std::uint64_t restored_host_bytes  = 0;

    [[nodiscard]] friend constexpr bool
    operator==(const MaterializationDiagnostics&,
               const MaterializationDiagnostics&) noexcept = default;
};

struct NgramArchiveStats {
    bool enabled              = false;
    bool bound                = false;
    bool published            = false;
    std::uint64_t generation  = 0;
    std::size_t sources       = 0;
    std::size_t session_bytes = 0;
    std::size_t total_bytes   = 0;
    // Effective seed after request-domain separation; archive state is also needed for replay.
    std::optional<std::uint64_t> sampling_seed;
};

struct GenerationResult {
    PromptSummary prompt;
    std::vector<TokenId> generated_token_ids;
    std::string content;
    std::string reasoning;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics tool_call_parse;
    std::uint32_t reasoning_tokens = 0;
    FinishReason finish_reason     = FinishReason::None;
    std::optional<std::string> matched_stop_string;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::Root;
    MaterializationDiagnostics materialization;
    GenerationTimings timings;
    GenerationEngineTiming engine_timing;
    SpeculativeStats speculative;
    NgramArchiveStats ngram_archive;
    ThinkingBudgetStats thinking;
};

struct ArenaMemorySummary {
    std::size_t capacity_bytes  = 0;
    std::size_t used_bytes      = 0;
    std::size_t peak_used_bytes = 0;
};

// Logical regions within the one physical workspace allocation. These byte values describe
// layout and live extents and must not be added to workspace.capacity_bytes.
struct VisionWorkspaceMemorySummary {
    std::uint32_t aggregate_prompt_tokens = 0;
    std::uint32_t max_item_tokens         = 0;
    std::size_t general_capacity_bytes    = 0;
    std::size_t encode_peak_bytes         = 0;
    std::size_t handoff_offset_bytes      = 0;
    std::size_t handoff_capacity_bytes    = 0;
    std::size_t handoff_active_bytes      = 0;
    std::size_t handoff_peak_bytes        = 0;
};

// Most tokens Engine::score_tokens returns per scored position for each distribution output.
inline constexpr std::uint32_t kMaximumScoreTopTokens = 64;

// Distribution outputs of Engine::score_tokens beyond every target token's log-probability. All
// log-probabilities are over the model's public tokens.
struct ScoreOptions {
    // The top_k most probable tokens of every scored position, in [0,64].
    std::uint32_t top_k = 0;
    // The log-probabilities of candidates_per_position given tokens at every scored position, in
    // [0,64]: candidates holds them position-major (scored positions x candidates_per_position).
    std::uint32_t candidates_per_position = 0;
    std::vector<TokenId> candidates;
};

struct ScoreResult {
    // log p(tokens[i] | tokens[0..i)) for every scored position i.
    std::vector<float> logprobs;
    // Position-major (scored positions x top_k): most probable first, equal logits by ascending
    // token id.
    std::vector<TokenId> top_ids;
    std::vector<float> top_logprobs;
    // Position-major (scored positions x candidates_per_position), in the candidates' order.
    std::vector<float> candidate_logprobs;
};

struct MemorySummary {
    int device                                = 0;
    std::uint32_t max_context                 = 0;
    KvCapacityMode kv_capacity_mode           = KvCapacityMode::Explicit;
    std::uint32_t kv_capacity                 = 0; // Resolved page-aligned Main KV capacity.
    std::uint32_t kv_capacity_page_groups     = 0;
    std::uint32_t kv_capacity_max_page_groups = 0;
    KvCacheStorage kv_cache                   = KvCacheStorage::BFloat16;
    ArenaMemorySummary weights;
    ArenaMemorySummary sequence;
    ArenaMemorySummary workspace;
    std::optional<VisionWorkspaceMemorySummary> vision_workspace;
    std::size_t minimum_runtime_reservation_bytes = 0;
    std::size_t kv_capacity_increment_bytes       = 0;
    std::size_t runtime_reservation_bytes         = 0;
    std::size_t available_after_weights_bytes     = 0;
    std::size_t available_after_startup_bytes     = 0;
    std::size_t kv_capacity_headroom_bytes        = 0;
    std::size_t planned_slack_bytes               = 0;
    std::size_t workspace_logical_peak_bytes      = 0;
    std::size_t cuda_graph_allowance_bytes        = 0;
    // Device memory CUDA Graph preparation actually took at startup (free memory before minus
    // after instantiating, uploading and launching every executable); 0 without CUDA Graphs.
    std::size_t cuda_graph_measured_bytes         = 0;
    std::size_t kv_payload_bytes                  = 0;
    std::uint32_t host_state_capacity_slots       = 0;
    std::uint32_t host_state_occupied_slots       = 0;
    std::size_t host_kv_capacity_bytes            = 0;
    std::size_t host_kv_occupied_bytes            = 0;
    // Host retention-tier unit costs. Every Host StateImage slot pins image_bytes regardless of
    // the prefix depth it resumes, and Host KV pins bytes per page group, so the split of a
    // budget between the two is the depth-versus-positions trade made visible.
    std::size_t host_state_image_bytes            = 0;
    std::size_t host_kv_page_group_bytes          = 0;
    // Engaged only when the single host RAM budget mode is active.
    std::size_t host_cache_budget_bytes           = 0;
};

// Worker-owned monotonic nanosecond counters. Top-level Host phases are mutually exclusive;
// device_wait_ns is blocked wall time and is intentionally excluded from their sum. Detail values
// are subsets of a top-level phase and must not be added to Host-active time again.
struct RuntimeHostWorkStats {
    std::uint64_t engine_boundary_ns      = 0;
    std::uint64_t program_submit_ns       = 0;
    std::uint64_t program_post_ns         = 0;
    std::uint64_t engine_commit_output_ns = 0;
    std::uint64_t engine_maintenance_ns   = 0;
    std::uint64_t device_wait_ns          = 0;

    std::uint64_t decode_host_ns         = 0;
    std::uint64_t decode_device_wait_ns  = 0;
    std::uint64_t prefill_host_ns        = 0;
    std::uint64_t prefill_device_wait_ns = 0;
    std::uint64_t control_host_ns        = 0;
    std::uint64_t control_device_wait_ns = 0;
    std::uint64_t prefill_units          = 0;
    std::uint64_t control_units          = 0;

    std::uint64_t admission_policy_ns           = 0;
    std::uint64_t context_progress_ns           = 0;
    std::uint64_t stats_publication_ns          = 0;
    std::uint64_t admission_policy_invocations  = 0;
    std::uint64_t context_progress_invocations  = 0;
    std::uint64_t stats_publication_invocations = 0;
};

// Monotonic execution counters, boundary-consistent current gauges, and explicitly named last
// decision observations. Consumers derive interval counters by subtracting two snapshots.
struct RuntimeStats {
    RuntimeHostWorkStats host_work;
    // Actual prompt tokens evaluated by prefill; reused checkpoint-prefix tokens are excluded.
    std::uint64_t computed_prefill_tokens = 0;
    // Tokens committed by decode rounds; the first token emitted by prefill is excluded.
    std::uint64_t committed_decode_tokens = 0;
    // Decode batch executions and the sum of their batch sizes.
    std::uint64_t decode_rounds             = 0;
    std::uint64_t decode_row_rounds         = 0;
    std::uint32_t running_requests          = 0;
    std::uint32_t prefilling_requests       = 0;
    std::uint32_t decode_ready_requests     = 0;
    std::uint32_t waiting_requests          = 0;
    std::uint32_t materializing_requests    = 0;
    std::uint32_t capture_pending_requests  = 0;
    std::uint32_t terminal_pending_requests = 0;
    std::uint64_t active_captures_completed = 0;
    std::uint64_t active_captures_aborted   = 0;
    // A capture the Program declined because the offer was not physically feasible. Unlike an
    // abort this is a silent retention loss, so it needs its own counter to be observable.
    std::uint64_t active_captures_skipped   = 0;

    std::uint64_t root_selections                    = 0;
    std::uint64_t private_endpoint_selections        = 0;
    std::uint64_t private_turn_closure_selections    = 0;
    std::uint64_t private_response_replay_selections = 0;
    std::uint64_t private_long_anchor_selections     = 0;
    std::uint64_t shared_stable_prefix_selections    = 0;
    std::uint64_t reused_prompt_tokens               = 0;
    std::uint32_t last_selected_frontier_tokens      = 0;
    // Aborted requests whose live state was published as a continuation endpoint.
    std::uint64_t salvaged_continuations             = 0;

    std::uint64_t state_moves     = 0;
    std::uint64_t state_forks     = 0;
    std::uint64_t state_restores  = 0;
    std::uint64_t state_d2h_count = 0;
    std::uint64_t state_h2d_count = 0;
    std::uint64_t state_d2d_count = 0;
    std::uint64_t state_d2h_bytes = 0;
    std::uint64_t state_h2d_bytes = 0;
    std::uint64_t state_d2d_bytes = 0;
    double state_d2h_seconds      = 0.0;
    double state_h2d_seconds      = 0.0;
    double state_d2d_seconds      = 0.0;

    std::uint64_t main_kv_d2h_pages    = 0;
    std::uint64_t main_kv_h2d_pages    = 0;
    std::uint64_t main_kv_d2d_pages    = 0;
    std::uint64_t main_kv_d2h_bytes    = 0;
    std::uint64_t main_kv_h2d_bytes    = 0;
    std::uint64_t main_kv_d2d_bytes    = 0;
    double main_kv_d2h_seconds         = 0.0;
    double main_kv_h2d_seconds         = 0.0;
    double main_kv_d2d_seconds         = 0.0;
    std::uint64_t backend_kv_d2h_pages = 0;
    std::uint64_t backend_kv_h2d_pages = 0;
    std::uint64_t backend_kv_d2d_pages = 0;
    std::uint64_t backend_kv_d2h_bytes = 0;
    std::uint64_t backend_kv_h2d_bytes = 0;
    std::uint64_t backend_kv_d2d_bytes = 0;
    double backend_kv_d2h_seconds      = 0.0;
    double backend_kv_h2d_seconds      = 0.0;
    double backend_kv_d2d_seconds      = 0.0;

    std::uint64_t pressure_spill_pages                 = 0;
    std::uint64_t partial_tail_cow_pages               = 0;
    std::uint32_t device_state_occupied_slots          = 0;
    std::uint32_t host_state_occupied_slots            = 0;
    std::uint32_t device_main_kv_occupied_pages        = 0;
    std::uint32_t device_backend_kv_occupied_pages     = 0;
    // Un-written growth reservation held by active requests, split out of the occupied totals
    // above. Occupancy alone cannot distinguish KV that exists from KV a request is merely still
    // entitled to, which is what made context-cache starvation invisible in the request log.
    std::uint32_t device_main_kv_lease_pages           = 0;
    std::uint32_t device_backend_kv_lease_pages        = 0;
    std::size_t host_kv_occupied_bytes                 = 0;
    std::uint64_t pressure_private_owners_degraded     = 0;
    std::uint64_t pressure_private_owners_evicted      = 0;
    std::uint64_t pressure_shared_owners_degraded      = 0;
    std::uint64_t pressure_shared_owners_evicted       = 0;
    std::uint64_t pressure_checkpoints_dropped         = 0;
    std::uint64_t pressure_searches                    = 0;
    std::uint64_t pressure_search_budget_exhaustions   = 0;
    std::uint64_t pressure_maximal_fallback_selections = 0;
    std::uint32_t shared_active_references             = 0;
    std::uint64_t historical_fork_hits                 = 0;
    double actual_context_transfer_seconds             = 0.0;

    // Hybrid prefix cache (ContextCacheMode::Hybrid); zero in Legacy mode. Block and snapshot
    // gauges are absolute; the rest are cumulative event counters.
    std::uint32_t hybrid_cached_blocks           = 0; // Device-resident tree blocks
    std::uint32_t hybrid_evictable_blocks        = 0;
    std::uint32_t hybrid_tree_blocks             = 0; // Device or Host
    std::uint32_t hybrid_snapshots               = 0;
    std::uint64_t hybrid_host_capacity_bytes     = 0;
    std::uint64_t hybrid_host_used_bytes         = 0;
    std::uint64_t hybrid_snapshot_hits           = 0;
    std::uint64_t hybrid_reused_tokens           = 0;
    std::uint64_t hybrid_blocks_inserted         = 0;
    std::uint64_t hybrid_blocks_reattached       = 0;
    std::uint64_t hybrid_blocks_duplicate        = 0;
    std::uint64_t hybrid_taps_created            = 0;
    std::uint64_t hybrid_taps_skipped            = 0;
    std::uint64_t hybrid_endpoints_created       = 0;
    std::uint64_t hybrid_host_image_writes       = 0;
    std::uint64_t hybrid_host_block_writes       = 0;
    std::uint64_t hybrid_host_image_restores     = 0;
    std::uint64_t hybrid_host_block_restores     = 0;
    std::uint64_t hybrid_host_write_bytes        = 0;
    std::uint64_t hybrid_host_restore_bytes      = 0;
    std::uint64_t hybrid_evicted_blocks          = 0;
    std::uint64_t hybrid_host_snapshot_evictions = 0;
    std::uint64_t hybrid_host_dead_reclaims      = 0;
    std::uint64_t hybrid_unbacked_node_losses    = 0;
};

enum class ContextCostPresetSource : std::uint8_t {
    GenericDefault,
    CompiledDefault,
    External,
};

[[nodiscard]] inline constexpr const char*
context_cost_preset_source_name(ContextCostPresetSource source) noexcept {
    switch (source) {
    case ContextCostPresetSource::GenericDefault:
        return "generic-default";
    case ContextCostPresetSource::CompiledDefault:
        return "compiled-default";
    case ContextCostPresetSource::External:
        return "external";
    }
    return "unknown";
}

struct ContextCostSummary {
    ContextCostPresetSource transfer_source = ContextCostPresetSource::GenericDefault;
    ContextCostPresetSource prefill_source  = ContextCostPresetSource::GenericDefault;
    std::string hardware_class;
    std::string prefill_signature;
    std::filesystem::path preset_path;
};

// Static facts about the loaded model, independent of the current request context and memory
// layout. The Model supplies the architecture dimension facts (vocab_size, embedding_size,
// native_context) and its registered identity; the Engine adds the artifact-measured facts
// (parameters, weight_bytes). Serving renders these into the OpenAI /v1/models model object and
// its llama.cpp-compatible `meta` field. The effective per-request context ceiling is a property
// of the running instance, not of the model, and is reported separately (meta n_ctx).
struct ModelMetadata {
    std::string model_id;    // Registered model identity (serving public alias default).
    std::string weights_id;  // Dominant loaded weights format name (meta ftype).
    std::uint64_t vocab_size     = 0; // Tokenizer token domain (meta n_vocab).
    std::uint64_t embedding_size = 0; // Model embedding width (meta n_embd).
    std::uint64_t native_context = 0; // Model native/training context (meta n_ctx_train).
    std::uint64_t parameters     = 0; // Total logical weight elements (meta n_params).
    std::uint64_t weight_bytes   = 0; // Encoded weight payload bytes (meta size).
};

struct LoadSummary {
    std::string architecture;
    std::string model_name;
    std::string cuda_sync_mode;
    std::vector<std::string> weight_formats;
    std::string prefill_signature;
    double load_seconds                = 0.0;
    double upload_seconds              = 0.0;
    std::uint64_t artifact_bytes_read  = 0;
    std::uint64_t host_to_device_bytes = 0;
    std::uint64_t peak_staging_bytes   = 0;
    std::size_t device_object_count    = 0;
    std::size_t host_object_count      = 0;
    ContextCostSummary context_cost;

    // Hybrid prefix cache restored from its persistent file at startup.
    struct PrefixCacheRestore {
        bool attempted = false;
        bool restored  = false;
        // Why nothing was restored (no file yet, incompatible file, I/O error).
        std::string message;
        std::uint64_t blocks    = 0;
        std::uint64_t snapshots = 0;
        std::uint64_t bytes     = 0;
        double seconds          = 0.0;
        // What the file holds and the Host tier bytes all of it takes, against this Engine's
        // tier. When the tier is smaller, only the snapshots it values most were restored.
        std::uint64_t saved_blocks        = 0;
        std::uint64_t saved_snapshots     = 0;
        std::uint64_t required_host_bytes = 0;
        std::uint64_t host_bytes          = 0;
    } prefix_cache;
};

} // namespace ninfer
