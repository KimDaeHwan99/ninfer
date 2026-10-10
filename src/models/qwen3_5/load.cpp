#include "models/qwen3_5/load.h"

#include "artifact/reader.h"
#include "models/qwen3_5/load/bindings.h"
#include "models/qwen3_5/load/vision_overlay.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ninfer::models::qwen3_5 {

// Evictable ranks: higher ranks sit closer to the arena end and are evicted first, so the
// low-traffic endpoints (MTP, draft head, embedding, LM head) form the borrowable tail that
// an overlay vision window streams the vision tower through.
constexpr std::uint32_t kEvictRankMtp        = 400;
constexpr std::uint32_t kEvictRankDraftHead  = 500;
constexpr std::uint32_t kEvictRankEmbedding  = 600;
constexpr std::uint32_t kEvictRankLmHead     = 700;

std::uint32_t overlay_evict_rank(std::string_view name) {
    if (name == "text/token_embedding") { return kEvictRankEmbedding; }
    if (name == "text/output_head") { return kEvictRankLmHead; }
    if (name == "proposal/head") { return kEvictRankDraftHead; }
    if (name.rfind("mtp/", 0) == 0) { return kEvictRankMtp; }
    return 0;
}

namespace {

// Whether two logical parameters read any stored object in common.
bool shares_object(const artifact::Reader& reader, std::string_view first,
                   std::string_view second) {
    const auto& bindings = reader.directory().bindings;
    const auto a         = bindings.find(first);
    const auto b         = bindings.find(second);
    if (a == bindings.end() || b == bindings.end()) { return false; }
    for (const auto& x : a->second.parts) {
        for (const auto& y : b->second.parts) {
            if (x.object == y.object) { return true; }
        }
    }
    return false;
}

} // namespace

struct LoadPlan::Impl {
    Config config;
    LoadOptions options;
    ModelWeights weights;
    std::vector<loading::PendingWeight> pending;
    artifact::MaterializationPlan materialization;
    std::optional<VisionOverlayLayout> vision_overlay;
    FrontendResources resources;
    InstanceInfo info;
    // The token embedding is HostPinned and read by kernels through its host address.
    bool host_embedding = false;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const Config& LoadPlan::config() const { return impl_->config; }

const ModelWeights& LoadPlan::weights() const { return impl_->weights; }

const FrontendResources& LoadPlan::resources() const { return impl_->resources; }

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    return impl_->materialization;
}

std::size_t LoadPlan::parameter_count() const { return impl_->pending.size(); }
const std::optional<VisionOverlayLayout>& LoadPlan::vision_overlay_layout() const {
    return impl_->vision_overlay;
}

const artifact::ParameterReference& LoadPlan::parameter(WeightId id) const {
    return impl_->pending.at(id.index).reference;
}

std::span<const WeightUse> LoadPlan::uses(WeightId id) const {
    return impl_->pending.at(id.index).uses;
}

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options,
                   TensorParallelPlacement placement) {
    if (options.embedding_offload == EmbeddingOffload::On && options.overlay_vision()) {
        // The overlay's eviction ladder borrows the device-resident embedding.
        throw std::invalid_argument(
            "--embedding-offload on is incompatible with --vision-offload on");
    }
    auto out     = std::make_unique<LoadPlan::Impl>();
    out->options = options;
    out->config  = parse_config(reader.directory(), options);

    // A table tied to the output head keeps its Device parent for the head, so pinning it as
    // well would only add system RAM.
    out->host_embedding = options.host_embedding(placement.split()) &&
                          !shares_object(reader, "text/token_embedding", "text/output_head");
    artifact::Binder binder(reader);
    out->resources = loading::bind_resources(binder, out->config);
    loading::Bindings bindings(binder);
    const auto& text  = out->config.text;
    out->weights.text = loading::bind_text(
        bindings, text, options,
        out->host_embedding ? artifact::Residency::HostPinned : artifact::Residency::Device);
    if (out->config.vision) {
        out->weights.vision = loading::bind_vision(bindings, *out->config.vision, text, options);
    }
    if (out->config.mtp) {
        out->weights.mtp = loading::bind_mtp(bindings, text, out->weights.text);
    }
    if (out->config.draft) {
        out->weights.draft =
            loading::bind_draft(bindings, *out->config.draft, text, out->weights.text,
                                std::string(options.speculative_component()));
    }
    if (options.proposal_enabled()) {
        const auto& proposal = reader.directory().component("text").proposal;
        if (!proposal) {
            throw artifact::ArtifactError("selected proposal head is absent from artifact");
        }
        out->weights.proposal = loading::bind_proposal(bindings, *proposal, text, options,
                                                       out->resources.public_token_count);
        if (out->config.draft && out->config.draft->dflash2) {
            const auto domain =
                proposal->indexed ? proposal->rows : out->resources.public_token_count;
            if (out->config.draft->dflash2->selector_top_k > domain) {
                throw artifact::ArtifactError(
                    "proposal domain is smaller than DFlash2 selector_top_k");
            }
        }
        if (out->weights.mtp) { out->weights.mtp->output_head = out->weights.proposal->head; }
        if (out->weights.draft) { out->weights.draft->output_head = out->weights.proposal->head; }
    }
    out->weights.text.output_head_use =
        bindings.use(out->weights.text.output_head, "text/final_hidden");
    if (out->weights.mtp) {
        out->weights.mtp->output_head_use =
            bindings.use(out->weights.mtp->output_head, "mtp/final_hidden");
    }
    if (out->weights.draft) {
        out->weights.draft->output_head_use =
            bindings.use(out->weights.draft->output_head,
                         std::string(options.speculative_component()) + "/final_hidden");
    }
    out->pending = std::move(bindings.weights);
    if (options.overlay_vision()) {
        if (placement.split()) {
            throw std::invalid_argument("--vision-offload is not supported at --tp 2");
        }
        for (auto& pending : out->pending) {
            const auto rank = overlay_evict_rank(pending.reference.name);
            if (rank == 0) { continue; }
            for (const auto& part : pending.reference.binding.parts) {
                binder.mark_device_evictable(part.object, rank);
            }
        }
    }
    out->materialization = std::move(binder).finish(
        options.overlay_vision() ? EvictableWeightPool::kChunkBytes : 1);
    if (options.overlay_vision()) {
        out->vision_overlay =
            compute_vision_overlay_layout(out->weights, out->pending, out->materialization);
    }
    if (placement.split()) {
        const Config full = out->config;
        out->config       = loading::shard_config(full, placement);
        loading::shard_plan(reader, full, out->pending, out->materialization, placement);
    }
    out->info.name = reader.directory().metadata.value(
        "name", std::string(architecture_name(text.architecture)));
    out->info.metadata_json   = reader.directory().metadata.dump();
    out->info.provenance_json = reader.directory().provenance.dump();
    out->info.artifact_id     = reader.artifact_id();
    return LoadPlan(std::move(out));
}

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                         const StartupObserver* observer) {
    if (!plan.impl_) { throw artifact::ArtifactError("load plan was already consumed"); }
    auto data = std::move(plan.impl_);
    std::unique_ptr<EvictableWeightPool> pool;
    if (data->options.overlay_vision()) {
        const auto& layout = *data->vision_overlay;
        if (layout.staging_bytes == 0) {
            throw std::logic_error("overlay vision requires a staged overlay window");
        }
        if (data->materialization.evictable_tail_bytes == 0) {
            throw std::logic_error("overlay vision requires an evictable weight ladder");
        }
        if (!EvictableWeightPool::supported(device.device)) {
            throw std::invalid_argument(
                "this device does not support virtual memory management, so "
                "--vision-offload on cannot be used; select off");
        }
        pool = std::make_unique<EvictableWeightPool>(EvictableWeightPool::Config{
            .arena_bytes          = data->materialization.device_capacity_bytes,
            .evictable_tail_bytes = data->materialization.evictable_tail_bytes,
            .overlay_bytes        = layout.staging_bytes,
            .device               = device.device});
    }
    auto backing = artifact::materialize(*data->materialization.source,
                                         std::move(data->materialization), device, observer,
                                         std::move(pool));
    if (data->host_embedding) {
        // The embedding gather dereferences the table's host address on the device, which is
        // valid only where UVA maps the pinned block at the same address.
        const auto pinned = backing.pinned_bytes_range();
        void* mapped      = nullptr;

        const cudaError_t status =
            pinned.empty() ? cudaErrorInvalidValue
                           : cudaHostGetDevicePointer(
                                 &mapped, const_cast<std::byte*>(pinned.data()), 0);
        if (status != cudaSuccess || mapped != pinned.data()) {
            (void)cudaGetLastError();
            throw std::runtime_error(
                "--embedding-offload on requires pinned memory mapped at the same address on the "
                "device; select off");
        }
    }
    std::optional<VisionOverlayAssets> overlay;
    if (data->options.overlay_vision()) {
        EvictableWeightPool* eviction = backing.eviction_pool();
        if (eviction == nullptr) {
            throw std::logic_error("overlay vision lost its evictable pool");
        }
        eviction->capture_tail_mirror(device.transfer_stream);
        const auto pinned = backing.pinned_bytes_range();
        if (pinned.empty()) {
            throw std::logic_error("overlay vision requires a pinned materialization");
        }
        overlay = VisionOverlayAssets{
            .pool         = eviction,
            .pinned_block = pinned.data(),
            .pinned_bytes = pinned.size(),
            .ladder_bytes = eviction->evictable_tail_bytes(),
            .layout       = std::move(*data->vision_overlay)};
    }
    auto bound = loading::resolve_weights(std::move(data->pending), backing);
    return std::unique_ptr<Model>(new Model(
        std::move(data->config), data->options, std::move(data->weights), std::move(bound),
        std::move(data->resources), std::move(data->info), std::move(overlay),
        std::move(backing)));
}

std::unique_ptr<Model> load_model(const std::filesystem::path& path, LoadOptions options,
                                  DeviceContext& device, const StartupObserver* observer) {
    artifact::Reader reader(path);
    return materialize_model(plan_load(reader, options), device, observer);
}

} // namespace ninfer::models::qwen3_5
