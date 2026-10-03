#include "serve/serve_options.h"
#include "product/device_placement_options.h"
#include "product/speculative_options.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace ninfer::serve {
namespace {

int parse_nonnegative_int(const char* text, const char* label) {
    char* end        = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 ||
        value > static_cast<long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<int>(value);
}

float parse_float_in(const char* text, const char* label, float lo, float hi) {
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !(value >= lo) || !(value <= hi)) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<float>(value);
}

std::uint64_t parse_u64(const char* text, const char* label) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " +
                                    (text == nullptr ? "" : text));
    }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

KvCacheStorage parse_kv_dtype(const char* text) {
    const std::string value(text);
    if (value == "bf16") { return KvCacheStorage::BFloat16; }
    if (value == "int8") { return KvCacheStorage::Int8Group64; }
    if (value == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    if (value == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (value == "k8v4") { return KvCacheStorage::Fp8KeyNvfp4Value; }
    throw std::invalid_argument("invalid kv-dtype: " + value);
}

KvCapacityPolicy parse_kv_capacity(const char* text) {
    if (std::string_view(text) == "auto") { return KvCapacityPolicy::automatic(); }
    const int value = parse_nonnegative_int(text, "kv-capacity");
    if (value == 0) { throw std::invalid_argument("--kv-capacity must be positive"); }
    return KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(value));
}

} // namespace

std::string serve_usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " <model.ninfer> [options]\n"
           "\n"
           "Serves the OpenAI Responses/Chat Completions and Anthropic Messages APIs.\n"
           "  --help, -h                 show this help and exit\n"
           "\n"
           "MODEL & CONTEXT\n"
           "  --max-context N            max context tokens per request (default 8192)\n"
           "  --max-concurrency N        max concurrent sequences, 1-8 (default 1)\n"
           "  --tp 1|2                   tensor-parallel width (default 1); 2 splits the model\n"
           "                             across two GPUs\n"
           "  --devices A,B              one CUDA device per rank, required by --tp 2\n"
           "                             (--device, when also given, must name the first)\n"
           "  --prefill-chunk N          prefill chunk size in tokens, multiple of 128\n"
           "                             (default 1024)\n"
           "  --use-original-int8-prefill-kernel\n"
           "                             prefill INT8 KV with the original prompt kernel at\n"
           "                             the requested chunk (default: the fast kernel, chunk\n"
           "                             rounded down to whole attention waves)\n"
           "  --use-original-nvfp4-prefill-kernel\n"
           "                             prefill NVFP4 KV with the tiled prompt kernel\n"
           "                             (default: the fast kernel)\n"
           "  --no-cuda-graph            disable CUDA-graph decode rounds (on by default)\n"
           "  --default-max-tokens N     default max_tokens when a request omits it\n"
           "                             (default " +
           std::to_string(kDefaultMaxTokens) +
           ")\n"
           "  --default-thinking-budget N  cap model-origin thinking for enabled\n"
           "                             requests; control tokens count toward the\n"
           "                             request output limit\n"
           "  --thinking-budget-message S  message fed to the model when it hits its thinking\n"
           "                             budget, replacing the built-in end-of-thinking notice\n"
           "                             (wrap the message in double quotes, e.g.\n"
           "                             --thinking-budget-message \"Time to stop thinking. I must "
           "act\n"
           "                             now:\")\n"
           "  --model-id ID              override the artifact metadata.name reported by\n"
           "                             the server\n"
           "  --chat-template FILE       replace the artifact frontend chat template at\n"
           "                             startup; must match a template the target accepts\n"
           "  --context-cost-presets F   runtime context-cost preset file (overrides\n"
           "                             matching compiled-in values)\n"
           "  --rope-yarn-factor F       runtime YaRN context extension factor, finite [1,4]\n"
           "                             (default 1); startup-fixed, extends the allowed\n"
           "                             ceiling only, not --max-context\n"
           "  --device N                 CUDA device ordinal (default 0)\n"
           "\n"
           "KV CACHE\n"
           "  --kv-capacity N|auto       KV-cache capacity in tokens (default auto, or\n"
           "                             --max-context with the original prefix caching\n"
           "                             system or --no-prefix-reuse; auto sizes to free\n"
           "                             VRAM, leaving " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           " MiB of headroom; configurable\n"
           "                             via --vram-headroom-mib)\n"
           "  --vram-headroom-mib N      VRAM headroom in MiB left by --kv-capacity auto\n"
           "                             (default " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           ")\n"
           "  --kv-dtype T               KV storage: bf16 (default) | int8 | fp8 | nvfp4 | k8v4\n"
           "  --no-prefix-reuse          disable prefix caching in either system below\n"
           "                             (enabled by default); cannot be combined with any\n"
           "                             prefix-cache option\n"
           "\n"
           "NEW PREFIX CACHING SYSTEM (hybrid; the default)\n"
           "  Content-addressed 64-token KV blocks shared across requests plus sparse model\n"
           "  state snapshots. It configures itself: free VRAM becomes Device block cache\n"
           "  (--kv-capacity defaults to auto) and every option below is optional.\n"
           "  --host-cache-mib N         pinned host RAM pool that KV blocks and state\n"
           "                             snapshots share (default 8192; 0 = GPU only)\n"
           "  --prefix-cache-file PATH   restore the host tier from PATH at startup when\n"
           "                             present (written by this binary for the same\n"
           "                             artifact and KV format) and save it there on clean\n"
           "                             shutdown (Ctrl+C twice); relative paths resolve\n"
           "                             against the launch directory (default off:\n"
           "                             nothing is saved)\n"
           "  --device-snapshot-slots N  device state snapshot slots (default concurrency\n"
           "                             + 1; + 2 without a host tier)\n"
           "  --cache-taps-per-request N new prefill snapshots per request (default 8;\n"
           "                             2 without a host tier)\n"
           "  --cache-tap-ladder N       ladder base in tokens for history snapshots\n"
           "                             (default max(4096, 2x prefill chunk))\n"
           "  --cache-tap-min-gap N      minimum tokens between ladder snapshots\n"
           "                             (default max(1024, prefill chunk))\n"
           "\n"
           "ORIGINAL PREFIX CACHING SYSTEM (upstream's checkpoint catalog, with fixes)\n"
           "  --use-original-prefix-caching\n"
           "                             use this system instead of the new one; the\n"
           "                             options below require it. --kv-capacity then\n"
           "                             defaults to --max-context\n"
           "  --host-cache-mib N         single host RAM ceiling: sizes the host state pool\n"
           "                             from the checkpoints the engine creates (state\n"
           "                             capped at half the budget), spends spare state\n"
           "                             room on more long anchors per continuation and\n"
           "                             gives host KV the rest; cannot be combined with\n"
           "                             --host-state-slots or --host-kv-mib\n"
           "  --device-state-slots N     extra device checkpoint slots beyond active lanes\n"
           "                             (default = --max-concurrency)\n"
           "  --host-state-slots N       host checkpoint slots (default 8)\n"
           "  --host-kv-mib N            host KV cache in MiB (default 8192)\n"
           "  --max-private-continuations N          bounded private catalogs\n"
           "                                         (default 2x concurrency)\n"
           "  --max-shared-prefixes N                bounded shared prefix catalogs\n"
           "                                         (default max(concurrency," +
           std::to_string(kMaximumPreparedPromptCacheCandidatesPerRequest) +
           "))\n"
           "  --max-long-anchors-per-continuation N  long anchors per continuation; the\n"
           "                                         engine anchors up to N message\n"
           "                                         boundaries (default 4; --host-cache-mib\n"
           "                                         raises it within budget)\n"
           "  --long-anchor-spacing N                minimum tokens between anchors,\n"
           "                                         doubling per anchor back from the\n"
           "                                         prompt end (default 1024; 0 anchors\n"
           "                                         every boundary)\n"
           "  defaults: device-state=max-concurrency, private=2x concurrency,\n"
           "  shared=max(concurrency," +
           std::to_string(kMaximumPreparedPromptCacheCandidatesPerRequest) +
           "), anchors=4; host state=8 slots, host KV=8192 MiB\n"
           "\n"
           "SPECULATIVE DECODING (off by default)\n"
           "  --spec mtp|dflash|dflash2    speculative decoding backend\n"
           "  --draft-tokens N           draft tokens per round (mtp 1-5; dflash/dflash2 1-15)\n"
           "  --lm-head-draft            use the optimized proposal head\n"
           "  --ngram-draft-tokens N     propose N verified ngram copies per round, 1-63 (0 off);\n"
           "                             above 15 requires --max-concurrency 1\n"
           "  --ngram-min-match N        minimum ngram match length, 4-64\n"
           "  --ngram-archive-mib N      MiB of retained source archive for ngram proposals\n"
           "  --ngram-session-mib N      MiB session-scoped ngram source budget (default 128);\n"
           "                             used only with --ngram-archive-mib\n"
           "  --ngram-native-sessions    retain ngram sources across compaction; requires\n"
           "                             --ngram-archive-mib\n"
           "\n"
           "VISION (off by default)\n"
           "  --vision                   enable media and load the Vision GPU allocations\n"
           "  --vision-offload on|off    keep the vision tower in pinned system RAM instead of\n"
           "                             VRAM (default off); on adds no steady-state VRAM and\n"
           "                             borrows device memory only while encoding an image;\n"
           "                             requires --vision\n"
           "  --vision-max-merged N      bound merged vision tokens, 64-32768 (default 32768)\n"
           "  --media-cache-mib N        retained decoded-media cache\n"
           "                             (default 1024; 0 disables)\n"
           "  --media-live-mib N         cap on live BF16 patch payloads (default 2048)\n"
           "  --media-preprocess-threads N  decode/preprocess workers (default 0 = auto,\n"
           "                             at most 16 workers; explicit max 64)\n"
           "\n"
           "SAMPLING (defaults come from the loaded model + thinking mode; server flags\n"
           "and request fields override individual values)\n"
           "  --temperature F            sampling temperature (0-2)\n"
           "  --top-p F                  nucleus probability (0-1)\n"
           "  --top-k N                  keep the top N tokens (0-20)\n"
           "  --min-p F                  minimum token probability (0-1)\n"
           "  --presence-penalty F       -2 to 2\n"
           "  --frequency-penalty F      -2 to 2\n"
           "  --seed N                   fixed random seed\n"
           "  --greedy                   force temperature 0 (exact argmax)\n"
           "  --no-thinking              disable the thinking mode (enabled by default)\n"
           "  --preserve-thinking        retain closed-turn assistant reasoning\n"
           "                             in later prompts\n"
           "  --tolerant-tool-calls      recover complete Qwen calls with malformed wrapper or\n"
           "                             suffix output, keep a final call cut by the output\n"
           "                             budget and an undeclared name (strict all-or-nothing\n"
           "                             by default)\n"
           "\n"
           "NETWORKING & RESOURCES\n"
           "  --host H                   listen address (default 127.0.0.1)\n"
           "  --port P                   listen port (default 8080)\n"
           "  --api-key KEY              require this key on every request (default: none)\n"
           "  --max-request-mib N        max request body in MiB (default 384; enforced\n"
           "                             pre-parse)\n"
           "  --max-pending-requests N   max queued requests (default 16)\n"
           "  --pending-timeout-ms N     queue timeout in ms (default 30000)\n"
           "  --request-log-jsonl FILE   append full-precision server/request records\n"
           "  --response-store-max-records N Responses-state record cap (default 1024)\n"
           "  --response-store-max-mib N      Responses-state byte cap in MiB (default 256)\n"
           "  --log-stats-interval-ms N  throughput-log interval in ms\n"
           "                             (default 5000; 0 disables)\n"
           "  --log-colours on|off       colour the console stats lines (default off; on\n"
           "                             colours the command-window log only, never file\n"
           "                             logs)\n"
           "  --log-stats-panel on|off   pin session averages (TTFT, cache hit, prefill,\n"
           "                             decode, MTP/DFlash and n-gram acceptance) beneath\n"
           "                             the console log (default on; interactive\n"
           "                             terminals only)\n"
           "  --log-level L              pretty stderr verbosity (trace|debug|info|warning|\n"
           "                             error|critical|off; default info)\n"
           "  --cors                     send permissive CORS headers for browser UIs\n"
           "  --usage-chunk-choice       give the streamed usage chunk a zero-delta choice so\n"
           "                             strict parsers that reject choices:[] accept it\n"
           "\n"
           "NOTES\n"
           "  --vram-headroom-mib requires --kv-capacity auto (the default with the new\n"
           "  prefix caching system).\n"
           "  Options of the two prefix caching systems cannot be mixed.\n"
           "  --vision-offload on requires --vision.\n"
           "  --ngram-draft-tokens above 15 requires --max-concurrency 1.\n"
           "  --ngram-native-sessions requires --ngram-archive-mib.\n"
           "  --rope-yarn-factor is startup-fixed, finite [1,4] (default 1); it extends the\n"
           "  allowed ceiling only, not --max-context.\n"
           "  sampler defaults come from the loaded model and resolved thinking mode;\n"
           "  server flags and request fields override individual values.\n"
           "  --greedy forces temperature 0 (exact argmax).\n"
           "  --tp 2 uses the original prefix caching system and does not support ngram\n"
           "  drafting.\n";
}

ServeOptions parse_serve_options(int argc, char** argv) {
    ServeOptions options;
    options.startup_argv.reserve(static_cast<std::size_t>(argc));
    bool redact_next = false;
    for (int i = 0; i < argc; ++i) {
        if (redact_next) {
            options.startup_argv.emplace_back("<redacted>");
            redact_next = false;
            continue;
        }
        options.startup_argv.emplace_back(argv[i] == nullptr ? "" : argv[i]);
        redact_next = options.startup_argv.back() == "--api-key";
    }
    bool default_max_tokens_explicit = false;
    std::optional<int> explicit_device;
    std::vector<int> rank_devices;
    std::uint32_t tensor_parallel = 1;
    bool kv_capacity_explicit        = false;
    bool context_capacity_explicit   = false;
    bool host_state_slots_explicit   = false;
    bool host_kv_mib_explicit        = false;
    bool host_cache_budget_explicit  = false;
    bool original_cache_selected     = false;
    // Last flag seen that belongs to only one prefix-cache mode, for the cross-mode error.
    const char* legacy_cache_flag  = nullptr;
    const char* hybrid_option_flag = nullptr;
    // The hybrid prefix cache is the server default; --use-original-prefix-caching selects Legacy.
    options.context_cache.mode = ContextCacheMode::Hybrid;
    std::optional<std::size_t> vram_headroom_mib;
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        options.help_requested = true;
        return options;
    }
    if (argc < 2) { throw std::invalid_argument("artifact path is required"); }
    options.artifact_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg    = argv[i];
        const auto require_value = [&](const char* flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };
        if (arg == "--host") {
            options.host = require_value("--host");
        } else if (arg == "--port") {
            options.port = parse_nonnegative_int(require_value("--port"), "port");
        } else if (arg == "--api-key") {
            options.api_key = require_value("--api-key");
        } else if (arg == "--model-id") {
            options.model_id_override = require_value("--model-id");
            if (options.model_id_override->empty()) {
                throw std::invalid_argument("--model-id must not be empty");
            }
        } else if (arg == "--chat-template") {
            options.chat_template_path = require_value("--chat-template");
            if (options.chat_template_path.empty()) {
                throw std::invalid_argument("--chat-template must not be empty");
            }
        } else if (arg == "--rope-yarn-factor") {
            options.rope_yarn_factor =
                parse_float_in(require_value("--rope-yarn-factor"), "rope-yarn-factor", 1.0F, 4.0F);
        } else if (arg == "--max-context") {
            options.max_context = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-context"), "max-context"));
        } else if (arg == "--kv-capacity") {
            options.kv_capacity  = parse_kv_capacity(require_value("--kv-capacity"));
            kv_capacity_explicit = true;
        } else if (arg == "--vram-headroom-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--vram-headroom-mib"), "vram-headroom-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--vram-headroom-mib is out of range");
            }
            vram_headroom_mib = static_cast<std::size_t>(mib);
        } else if (arg == "--max-concurrency") {
            options.max_concurrency = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-concurrency"), "max-concurrency"));
        } else if (arg == "--max-pending-requests") {
            options.max_pending_requests = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--max-pending-requests"), "max-pending-requests"));
        } else if (arg == "--pending-timeout-ms") {
            options.pending_timeout_ms = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--pending-timeout-ms"), "pending-timeout-ms"));
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--prefill-chunk"), "prefill-chunk"));
        } else if (arg == "--use-original-int8-prefill-kernel") {
            options.original_int8_prefill_kernel = true;
        } else if (arg == "--use-original-nvfp4-prefill-kernel") {
            options.original_nvfp4_prefill_kernel = true;
        } else if (arg == "--context-cost-presets") {
            options.context_cost_presets = require_value("--context-cost-presets");
            if (options.context_cost_presets.empty()) {
                throw std::invalid_argument("--context-cost-presets must not be empty");
            }
        } else if (arg == "--log-stats-interval-ms") {
            options.log_stats_interval_ms = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--log-stats-interval-ms"), "log-stats-interval-ms"));
        } else if (arg == "--log-colours") {
            const std::string_view value = require_value("--log-colours");
            if (value == "on") {
                options.log_colours = true;
            } else if (value == "off") {
                options.log_colours = false;
            } else {
                throw std::invalid_argument("--log-colours accepts on or off");
            }
        } else if (arg == "--log-stats-panel") {
            const std::string_view value = require_value("--log-stats-panel");
            if (value == "on") {
                options.log_stats_panel = true;
            } else if (value == "off") {
                options.log_stats_panel = false;
            } else {
                throw std::invalid_argument("--log-stats-panel accepts on or off");
            }
        } else if (arg == "--max-request-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--max-request-mib"), "max-request-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--max-request-mib is out of range");
            }
            options.max_request_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-cache-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--media-cache-mib"), "media-cache-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--media-cache-mib is out of range");
            }
            options.media_cache_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-live-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--media-live-mib"), "media-live-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--media-live-mib is out of range");
            }
            options.media_live_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-preprocess-threads") {
            const int threads = parse_nonnegative_int(require_value("--media-preprocess-threads"),
                                                      "media-preprocess-threads");
            if (threads > 64) {
                throw std::invalid_argument("--media-preprocess-threads must be in [0,64]");
            }
            options.media_preprocess_threads = static_cast<std::uint32_t>(threads);
        } else if (arg == "--use-original-prefix-caching") {
            options.context_cache.mode = ContextCacheMode::Legacy;
            original_cache_selected    = true;
        } else if (arg == "--device-snapshot-slots") {
            options.context_cache.hybrid.device_snapshot_slots =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--device-snapshot-slots"), "device-snapshot-slots"));
            hybrid_option_flag = "--device-snapshot-slots";
        } else if (arg == "--cache-taps-per-request") {
            options.context_cache.hybrid.max_new_taps =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--cache-taps-per-request"), "cache-taps-per-request"));
            hybrid_option_flag = "--cache-taps-per-request";
        } else if (arg == "--cache-tap-ladder") {
            options.context_cache.hybrid.tap_ladder_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--cache-tap-ladder"), "cache-tap-ladder"));
            hybrid_option_flag = "--cache-tap-ladder";
        } else if (arg == "--prefix-cache-file") {
            options.context_cache.hybrid.persistent_file = require_value("--prefix-cache-file");
            if (options.context_cache.hybrid.persistent_file.empty()) {
                throw std::invalid_argument("--prefix-cache-file must not be empty");
            }
            hybrid_option_flag = "--prefix-cache-file";
        } else if (arg == "--cache-tap-min-gap") {
            options.context_cache.hybrid.tap_min_gap_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--cache-tap-min-gap"), "cache-tap-min-gap"));
            hybrid_option_flag = "--cache-tap-min-gap";
        } else if (arg == "--device-state-slots") {
            legacy_cache_flag                        = "--device-state-slots";
            options.context_cache.device_state_slots = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--device-state-slots"), "device-state-slots"));
            context_capacity_explicit = true;
        } else if (arg == "--host-state-slots") {
            legacy_cache_flag                      = "--host-state-slots";
            options.context_cache.host_state_slots = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--host-state-slots"), "host-state-slots"));
            context_capacity_explicit = true;
            host_state_slots_explicit = true;
        } else if (arg == "--host-kv-mib") {
            legacy_cache_flag       = "--host-kv-mib";
            const std::uint64_t mib = parse_u64(require_value("--host-kv-mib"), "host-kv-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--host-kv-mib is out of range");
            }
            options.context_cache.host_kv_capacity_bytes = static_cast<std::size_t>(mib << 20);
            context_capacity_explicit                    = true;
            host_kv_mib_explicit                         = true;
        } else if (arg == "--host-cache-mib") {
            const std::uint64_t mib = parse_u64(require_value("--host-cache-mib"), "host-cache-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--host-cache-mib is out of range");
            }
            options.context_cache.host_cache_budget_bytes = static_cast<std::size_t>(mib << 20);
            context_capacity_explicit                     = true;
            host_cache_budget_explicit                    = true;
        } else if (arg == "--max-private-continuations") {
            legacy_cache_flag = "--max-private-continuations";
            options.context_cache.max_private_continuations =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--max-private-continuations"), "max-private-continuations"));
            context_capacity_explicit = true;
        } else if (arg == "--max-shared-prefixes") {
            legacy_cache_flag = "--max-shared-prefixes";
            options.context_cache.max_shared_prefixes =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--max-shared-prefixes"), "max-shared-prefixes"));
            context_capacity_explicit = true;
        } else if (arg == "--max-long-anchors-per-continuation") {
            legacy_cache_flag = "--max-long-anchors-per-continuation";
            options.context_cache.max_long_anchors_per_continuation = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-long-anchors-per-continuation"),
                                      "max-long-anchors-per-continuation"));
            context_capacity_explicit = true;
        } else if (arg == "--long-anchor-spacing") {
            legacy_cache_flag                                    = "--long-anchor-spacing";
            options.context_cache.long_anchor_min_spacing_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--long-anchor-spacing"),
                                      "long-anchor-spacing"));
        } else if (arg == "--request-log-jsonl") {
            options.request_log_jsonl = require_value("--request-log-jsonl");
            if (options.request_log_jsonl.empty()) {
                throw std::invalid_argument("--request-log-jsonl must not be empty");
            }
        } else if (arg == "--response-store-max-records") {
            const int records = parse_nonnegative_int(require_value("--response-store-max-records"),
                                                      "response-store-max-records");
            if (records == 0) {
                throw std::invalid_argument("--response-store-max-records must be positive");
            }
            options.response_store_max_records = static_cast<std::size_t>(records);
        } else if (arg == "--response-store-max-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--response-store-max-mib"), "response-store-max-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--response-store-max-mib is out of range");
            }
            options.response_store_max_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--device") {
            explicit_device = parse_nonnegative_int(require_value("--device"), "device");
        } else if (arg == "--tp") {
            tensor_parallel = product::parse_tensor_parallel(require_value("--tp"));
        } else if (arg == "--devices") {
            rank_devices = product::parse_device_list(require_value("--devices"));
        } else if (arg == "--kv-dtype") {
            options.kv_cache = parse_kv_dtype(require_value("--kv-dtype"));
        } else if (arg == "--spec") {
            options.speculative.backend =
                product::parse_speculative_backend(require_value("--spec"));
        } else if (arg == "--draft-tokens") {
            options.speculative.draft_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--draft-tokens"), "draft-tokens"));
        } else if (arg == "--ngram-draft-tokens") {
            options.speculative.ngram_draft_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--ngram-draft-tokens"), "ngram-draft-tokens"));
        } else if (arg == "--ngram-min-match") {
            options.speculative.ngram_min_match = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--ngram-min-match"), "ngram-min-match"));
        } else if (arg == "--ngram-archive-mib" || arg == "--ngram-session-mib") {
            const auto mib = parse_u64(require_value(arg.c_str()), arg.c_str());
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("ngram archive capacity is out of range");
            }
            auto& bytes = arg == "--ngram-archive-mib" ? options.speculative.ngram_archive_bytes
                                                       : options.speculative.ngram_session_bytes;
            bytes       = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--ngram-native-sessions") {
            options.ngram_native_sessions = true;
        } else if (arg == "--default-max-tokens") {
            options.default_max_tokens =
                parse_nonnegative_int(require_value("--default-max-tokens"), "default-max-tokens");
            default_max_tokens_explicit = true;
        } else if (arg == "--default-thinking-budget") {
            const std::uint64_t budget =
                parse_u64(require_value("--default-thinking-budget"), "default-thinking-budget");
            if (budget == 0 || budget > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("--default-thinking-budget is out of range");
            }
            options.default_thinking_budget = static_cast<std::uint32_t>(budget);
        } else if (arg == "--thinking-budget-message") {
            options.thinking_budget_message = require_value("--thinking-budget-message");
            if (options.thinking_budget_message.empty()) {
                throw std::invalid_argument("--thinking-budget-message must not be empty");
            }
        } else if (arg == "--vision") {
            options.enable_vision = true;
        } else if (arg == "--vision-offload") {
            const std::string_view value = require_value("--vision-offload");
            if (value == "on") {
                options.vision_offload = true;
            } else if (value == "off") {
                options.vision_offload = false;
            } else {
                throw std::invalid_argument("--vision-offload accepts on or off");
            }
        } else if (arg == "--vision-max-merged") {
            const std::uint32_t merged =
                parse_nonnegative_int(require_value("--vision-max-merged"), "vision-max-merged");
            if (merged < 64 || merged > 32768) {
                throw std::invalid_argument("--vision-max-merged must be in [64, 32768]");
            }
            options.vision_max_merged_tokens = merged;
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "--no-prefix-reuse") {
            options.allow_prefix_reuse = false;
        } else if (arg == "--lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Optimized;
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--preserve-thinking") {
            options.preserve_thinking = true;
        } else if (arg == "--tolerant-tool-calls") {
            options.tolerant_tool_calls = true;
        } else if (arg == "--cors") {
            options.enable_cors = true;
        } else if (arg == "--usage-chunk-choice") {
            options.usage_chunk_choice = true;
        } else if (arg == "--temperature") {
            options.sampling_overrides.temperature =
                parse_float_in(require_value("--temperature"), "temperature", 0.0f, 2.0f);
        } else if (arg == "--top-p") {
            options.sampling_overrides.top_p =
                parse_float_in(require_value("--top-p"), "top-p", 0.0f, 1.0f);
        } else if (arg == "--top-k") {
            const int top_k = parse_nonnegative_int(require_value("--top-k"), "top-k");
            if (top_k > 20) { throw std::invalid_argument("top-k must be in [0,20]"); }
            options.sampling_overrides.top_k = top_k;
        } else if (arg == "--min-p") {
            options.sampling_overrides.min_p =
                parse_float_in(require_value("--min-p"), "min-p", 0.0f, 1.0f);
        } else if (arg == "--presence-penalty") {
            options.sampling_overrides.presence_penalty = parse_float_in(
                require_value("--presence-penalty"), "presence-penalty", -2.0f, 2.0f);
        } else if (arg == "--frequency-penalty") {
            options.sampling_overrides.frequency_penalty = parse_float_in(
                require_value("--frequency-penalty"), "frequency-penalty", -2.0f, 2.0f);
        } else if (arg == "--seed") {
            options.sampling_overrides.seed = parse_u64(require_value("--seed"), "seed");
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else if (arg == "--log-level") {
            options.log_level = product::parse_log_level(require_value("--log-level"));
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    // The hybrid cache keeps per-rank state the tensor-parallel split does not mirror, so --tp 2
    // runs the original system.
    if (tensor_parallel == 2 && options.allow_prefix_reuse) {
        if (hybrid_option_flag != nullptr) {
            throw std::invalid_argument(std::string(hybrid_option_flag) +
                                        " configures the hybrid prefix cache, which --tp 2 does "
                                        "not support");
        }
        options.context_cache.mode = ContextCacheMode::Legacy;
        original_cache_selected    = true;
    }
    if (!kv_capacity_explicit) {
        // The hybrid cache turns every Device page no active request holds into block cache, so
        // it sizes the KV pool to free VRAM unless a capacity is given. Without a prefix cache
        // pages beyond the active requests would sit unused.
        options.kv_capacity =
            options.allow_prefix_reuse && options.context_cache.mode == ContextCacheMode::Hybrid
                ? KvCapacityPolicy::automatic()
                : KvCapacityPolicy::explicit_capacity(options.max_context);
    }
    if (vram_headroom_mib.has_value()) {
        if (options.kv_capacity.mode != KvCapacityMode::Automatic) {
            throw std::invalid_argument("--vram-headroom-mib requires --kv-capacity auto");
        }
        options.kv_capacity = KvCapacityPolicy::automatic(*vram_headroom_mib << 20);
    }
    options.devices = product::resolve_rank_devices(tensor_parallel, rank_devices, explicit_device);
    options.device  = options.devices.front();
    if (!options.allow_prefix_reuse) {
        if (original_cache_selected) {
            throw std::invalid_argument(
                "--use-original-prefix-caching cannot be combined with --no-prefix-reuse");
        }
        if (context_capacity_explicit || legacy_cache_flag != nullptr ||
            hybrid_option_flag != nullptr) {
            throw std::invalid_argument(
                "--no-prefix-reuse cannot be combined with prefix-cache options");
        }
        options.context_cache.enabled                = false;
        options.context_cache.mode                   = ContextCacheMode::Legacy;
        options.context_cache.host_state_slots       = 0;
        options.context_cache.host_kv_capacity_bytes = 0;
    } else if (options.context_cache.mode == ContextCacheMode::Hybrid) {
        if (legacy_cache_flag != nullptr) {
            throw std::invalid_argument(std::string(legacy_cache_flag) +
                                        " configures the original prefix cache and requires "
                                        "--use-original-prefix-caching");
        }
        std::filesystem::path& file = options.context_cache.hybrid.persistent_file;
        if (!file.empty()) {
            if (host_cache_budget_explicit && options.context_cache.host_cache_budget_bytes == 0) {
                throw std::invalid_argument(
                    "--prefix-cache-file saves the Host tier, which --host-cache-mib 0 removes");
            }
            // Resolved now, so the save at shutdown writes where startup read, and checked now,
            // so an unusable location fails at launch rather than after a session of caching.
            file = std::filesystem::absolute(file).lexically_normal();
            std::error_code error;
            if (std::filesystem::is_directory(file, error)) {
                throw std::invalid_argument("--prefix-cache-file " + file.string() +
                                            " is a directory; name a file in it");
            }
            if (!std::filesystem::is_directory(file.parent_path(), error)) {
                throw std::invalid_argument("--prefix-cache-file " + file.string() +
                                            ": the directory " + file.parent_path().string() +
                                            " does not exist");
            }
        }
    } else if (hybrid_option_flag != nullptr) {
        throw std::invalid_argument(std::string(hybrid_option_flag) +
                                    " configures the hybrid prefix cache and cannot be combined "
                                    "with --use-original-prefix-caching");
    }
    if (host_cache_budget_explicit) {
        // The budget is the one host RAM ceiling; the two component flags would silently
        // fight it, and their independent-allocation semantics are exactly what the budget
        // exists to replace.
        if (host_state_slots_explicit || host_kv_mib_explicit) {
            throw std::invalid_argument(
                "--host-cache-mib cannot be combined with --host-state-slots or --host-kv-mib: "
                "the budget derives both Host state slots and Host KV bytes");
        }
    }
    if (options.port <= 0 || options.port > 65535) {
        throw std::invalid_argument("--port must be in [1,65535]");
    }
    if (options.max_context == 0) { throw std::invalid_argument("--max-context must be positive"); }
    if (options.kv_capacity.mode == KvCapacityMode::Explicit &&
        options.kv_capacity.explicit_tokens < options.max_context) {
        throw std::invalid_argument("--kv-capacity must be at least --max-context");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("--max-concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0) {
        throw std::invalid_argument("--max-pending-requests must be positive");
    }
    if (options.pending_timeout_ms == 0) {
        throw std::invalid_argument("--pending-timeout-ms must be positive");
    }
    if (options.max_request_bytes == 0) {
        throw std::invalid_argument("--max-request-mib must be positive");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a positive multiple of 128");
    }
    product::validate_speculative_cli_options(options.speculative);
    if (options.vision_offload && !options.enable_vision) {
        throw std::invalid_argument("--vision-offload on requires --vision");
    }
    // A speculative decode frame is allocated at the wider of the neural and ngram draft windows
    // and cannot be narrowed for a multi-request batch. The GDN conv-record workspace admits at
    // most 16 verification columns when the batch holds more than one request, so a wider ngram
    // proposal is admitted only for a single active request.
    if (options.speculative.ngram_draft_tokens > 15 && options.max_concurrency != 1) {
        throw std::invalid_argument("--ngram-draft-tokens above 15 requires --max-concurrency 1");
    }
    if (options.ngram_native_sessions && options.speculative.ngram_archive_bytes == 0) {
        throw std::invalid_argument("--ngram-native-sessions requires --ngram-archive-mib");
    }
    if (default_max_tokens_explicit) {
        if (options.default_max_tokens <= 0) {
            throw std::invalid_argument("--default-max-tokens must be positive");
        }
    }
    return options;
}

std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_name) {
    if (options.model_id_override.has_value()) { return *options.model_id_override; }
    if (artifact_model_name.empty()) {
        throw std::logic_error("loaded artifact model name must not be empty");
    }
    return std::string(artifact_model_name);
}

} // namespace ninfer::serve
