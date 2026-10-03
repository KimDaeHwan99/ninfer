#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/peer_executor.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5 {


SequencePlan::SequencePlan(std::unique_ptr<detail::SequencePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}

SequencePlan::SequencePlan(SequencePlan&&) noexcept = default;

SequencePlan& SequencePlan::operator=(SequencePlan&&) noexcept = default;

SequencePlan::~SequencePlan() = default;

std::uint32_t SequencePlan::capacity() const noexcept {
    return impl_ != nullptr ? impl_->capacity : 0;
}

std::uint32_t SequencePlan::kv_capacity() const noexcept {
    return impl_ != nullptr ? impl_->kv_capacity : 0;
}

std::uint32_t SequencePlan::max_concurrency() const noexcept {
    return impl_ != nullptr ? impl_->max_concurrency : 0;
}

std::size_t SequencePlan::device_reservation_bytes() const noexcept {
    return impl_ != nullptr ? impl_->device_reservation_bytes : 0;
}

std::size_t SequencePlan::workspace_capacity_bytes() const noexcept {
    return impl_ != nullptr ? impl_->workspace.capacity : 0;
}

const ContextCacheOptions& SequencePlan::context_cache_options() const noexcept {
    static const ContextCacheOptions empty;
    return impl_ != nullptr ? impl_->context_cache : empty;
}

SequencePlanner::SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl> impl) noexcept
    : impl_(std::move(impl)) {}

SequencePlanner::SequencePlanner(SequencePlanner&&) noexcept = default;

SequencePlanner& SequencePlanner::operator=(SequencePlanner&&) noexcept = default;

SequencePlanner::~SequencePlanner() = default;

const runtime::SequenceCapacityCurve& SequencePlanner::capacity_curve() const noexcept {
    static const runtime::SequenceCapacityCurve empty;
    return impl_ != nullptr ? impl_->curve : empty;
}

SequencePlan SequencePlanner::finalize(std::uint32_t main_page_groups) && {
    if (impl_ == nullptr) { throw std::logic_error("sequence planner is empty"); }
    return SequencePlan(detail::finalize_sequence_plan_impl(std::move(impl_), main_page_groups));
}

RequestBasePlan::RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}

RequestBasePlan::RequestBasePlan(RequestBasePlan&&) noexcept = default;

RequestBasePlan& RequestBasePlan::operator=(RequestBasePlan&&) noexcept = default;

RequestBasePlan::~RequestBasePlan() = default;

const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const PreparedContextCache& RequestBasePlan::context_cache() const noexcept {
    static const PreparedContextCache empty;
    return impl_ != nullptr ? impl_->context_cache : empty;
}

std::optional<PrefixShortlistKey>
RequestBasePlan::prefix_shortlist_key(std::uint32_t frontier) const noexcept {
    if (impl_ == nullptr || frontier == 0 || frontier > impl_->prefix_digests.size()) {
        return std::nullopt;
    }
    return PrefixShortlistKey{
        .digests      = impl_->prefix_digests.at(frontier),
        .frontier     = frontier,
        .identity_tag = impl_->prefix_identity_tag,
    };
}

std::optional<runtime::PrefillWork>
RequestBasePlan::shared_candidate_rebuild_work(std::uint32_t frontier) const noexcept {
    if (impl_ == nullptr) { return std::nullopt; }
    const auto found = std::find_if(impl_->shared_candidates.begin(),
                                    impl_->shared_candidates.end(), [&](const auto& candidate) {
                                        return candidate.frontier == frontier && candidate.identity;
                                    });
    return found == impl_->shared_candidates.end()
               ? std::nullopt
               : std::optional<runtime::PrefillWork>(found->identity->rebuild_work);
}

PressurePlanningSession::PressurePlanningSession(
    std::unique_ptr<detail::PressurePlanningSessionImpl> impl) noexcept
    : impl_(std::move(impl)) {}

PressurePlanningSession::PressurePlanningSession(PressurePlanningSession&&) noexcept = default;

PressurePlanningSession&
PressurePlanningSession::operator=(PressurePlanningSession&&) noexcept = default;

PressurePlanningSession::~PressurePlanningSession() = default;

CapturePressurePlanningSession::CapturePressurePlanningSession(
    CapturePressurePlanningSession&&) noexcept = default;

CapturePressurePlanningSession&
CapturePressurePlanningSession::operator=(CapturePressurePlanningSession&&) noexcept = default;

CapturePressurePlanningSession::~CapturePressurePlanningSession() = default;

PressureTargetHandle
PressurePlanningSession::identity_target(runtime::PlanningCandidateId candidate) const {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->identity_target(candidate);
}

PressureTargetHandle
PressurePlanningSession::root_maximal_target(runtime::PlanningCandidateId root_candidate) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->root_maximal_target(root_candidate);
}

PressureTargetHandle
PressurePlanningSession::maximal_target(runtime::PlanningCandidateId candidate) {
    return impl_->maximal_target(candidate);
}

PressureTargetHandle PressurePlanningSession::recency_maximal_target(
    runtime::PlanningCandidateId candidate, std::uint32_t sacrifice_oldest,
    std::span<const std::uint32_t> spared_ranks, bool demote_kept) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->recency_maximal_target(candidate, sacrifice_oldest, spared_ranks, demote_kept);
}

std::uint32_t PressurePlanningSession::ranked_owner_count() const {
    if (impl_ == nullptr) { return 0; }
    return impl_->ranked_owner_count();
}

void PressurePlanningSession::set_eviction_licence(std::uint32_t oldest_licensed,
                                                   std::span<const std::uint32_t> spared_ranks) {
    if (impl_ != nullptr) { impl_->set_eviction_licence(oldest_licensed, spared_ranks); }
}

std::uint32_t PressurePlanningSession::optional_targets_remaining() const noexcept {
    return impl_ != nullptr ? impl_->optional_targets_remaining() : 0;
}

PressureConstructionCursor PressurePlanningSession::begin_construction(PressureTargetHandle target,
                                                                       bool restore) {
    return impl_->begin_construction(target, restore);
}

runtime::PressureConstructionStep
PressurePlanningSession::next_construction_option(PressureConstructionCursor& cursor) {
    return impl_->next_construction_option(cursor);
}

void PressurePlanningSession::choose_construction(PressureConstructionCursor& cursor,
                                                  runtime::PressureConstructionOptionId option) {
    impl_->choose_construction(cursor, option);
}

std::optional<PressureTargetHandle>
PressurePlanningSession::construction_target(const PressureConstructionCursor& cursor) {
    return impl_->construction_target(cursor);
}

runtime::PressureTargetGuidance PressurePlanningSession::guidance(PressureTargetHandle target) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->guidance(target);
}

AssessedPressureTarget PressurePlanningSession::assess(PressureTargetHandle target) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->assess(target);
}

PreparedPressureExpansion PressurePlanningSession::prepare_expansion(PressureTargetHandle parent,
                                                                     std::uint32_t maximum_owners) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->prepare_expansion(parent, maximum_owners);
}

PressureExpansionView
PressurePlanningSession::commit_expansion(PreparedPressureExpansion&& prepared) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->commit_expansion(std::move(prepared));
}

void PressurePlanningSession::discard_expansion(PreparedPressureExpansion&& prepared) noexcept {
    if (impl_ != nullptr) { impl_->discard_expansion(std::move(prepared)); }
}

runtime::PrefillWork PressurePlanningSession::shared_capture_split_prefill_work(
    const AssessedPressureTarget& assessed, const PreparedPrompt& prompt,
    std::span<const std::uint32_t> frontiers) const {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->shared_capture_split_prefill_work(assessed, PreparedPromptAccess::view(prompt),
                                                    frontiers);
}

std::optional<ResourcePlan> PressurePlanningSession::seal(AssessedPressureTarget&& assessed,
                                                          const PreparedPrompt& prompt,
                                                          runtime::FinalScheduleIntent intent) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    std::optional<AdmissionCandidate> sealed =
        impl_->seal(std::move(assessed), PreparedPromptAccess::view(prompt), intent);
    if (!sealed) { return std::nullopt; }
    const bool needs_transfer = sealed->impl_->needs_transfer;
    return ResourcePlan(std::move(*sealed), impl_->resource_revision, needs_transfer);
}

bool PressurePlanningSession::try_claim_seal_window() noexcept {
    return impl_ != nullptr && impl_->program->try_claim_seal_window();
}

void PressurePlanningSession::release_seal_window() noexcept {
    if (impl_ != nullptr) { impl_->program->release_seal_window(); }
}

std::optional<CapturePressurePlan>
PressurePlanningSession::seal_capture(AssessedPressureTarget&& assessed) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    std::optional<CapturePressureCandidate> sealed = impl_->seal_capture(std::move(assessed));
    if (!sealed) { return std::nullopt; }
    return CapturePressurePlan(std::move(*sealed), impl_->resource_revision);
}

PressureTargetHandle CapturePressurePlanningSession::identity_target() const {
    if (!candidate_.impl_) { throw std::logic_error("capture pressure candidate is empty"); }
    return session_.identity_target(candidate_id());
}

runtime::PressureTargetGuidance
CapturePressurePlanningSession::guidance(PressureTargetHandle target) {
    return session_.guidance(target);
}

AssessedPressureTarget CapturePressurePlanningSession::assess(PressureTargetHandle target) {
    return session_.assess(target);
}

std::uint32_t CapturePressurePlanningSession::optional_targets_remaining() const noexcept {
    return session_.optional_targets_remaining();
}

PreparedPressureExpansion
CapturePressurePlanningSession::prepare_expansion(PressureTargetHandle parent) {
    return session_.prepare_expansion(parent);
}

PressureExpansionView
CapturePressurePlanningSession::commit_expansion(PreparedPressureExpansion&& prepared) {
    return session_.commit_expansion(std::move(prepared));
}

void CapturePressurePlanningSession::discard_expansion(
    PreparedPressureExpansion&& prepared) noexcept {
    session_.discard_expansion(std::move(prepared));
}

std::optional<CapturePressurePlan>
CapturePressurePlanningSession::seal(AssessedPressureTarget&& assessed) {
    return session_.seal_capture(std::move(assessed));
}

namespace {

using Access = detail::RuntimeContractAccess;

[[noreturn]] void ranks_diverged(const char* what) {
    throw std::logic_error(std::string("tensor-parallel ranks diverged: ") + what);
}

void require_same(bool same, const char* what) {
    if (!same) { ranks_diverged(what); }
}

bool same_tokens(const PendingBatch& left, const PendingBatch& right) {
    return left.row_count() == right.row_count() && left.row_stride() == right.row_stride() &&
           std::equal(left.tokens().begin(), left.tokens().end(), right.tokens().begin(),
                      right.tokens().end()) &&
           std::equal(left.row_counts().begin(), left.row_counts().end(),
                      right.row_counts().begin(), right.row_counts().end());
}

} // namespace

bool CapturePressurePlanningSession::try_claim_seal_window() noexcept {
    return session_.try_claim_seal_window();
}

void CapturePressurePlanningSession::release_seal_window() noexcept {
    session_.release_seal_window();
}

Program::Program(std::unique_ptr<detail::ProgramImpl> impl) noexcept : impl_(std::move(impl)) {}

Program::Program(std::unique_ptr<detail::ProgramImpl> impl,
                 std::unique_ptr<detail::ProgramImpl> peer,
                 std::unique_ptr<detail::PeerExecutor> executor) noexcept
    : impl_(std::move(impl)), peer_(std::move(peer)), executor_(std::move(executor)) {}

Program::~Program() noexcept {
    if (peer_) {
        // Each rank releases its device state on its own device thread.
        on_ranks_noexcept([&] { peer_.reset(); }, [&] { impl_.reset(); });
    }
}

void Program::on_ranks(const std::function<void()>& peer, const std::function<void()>& local) {
    if (!peer_) {
        local();
        return;
    }
    executor_->run(peer, local);
}

void Program::on_ranks_noexcept(const std::function<void()>& peer,
                                const std::function<void()>& local) noexcept {
    try {
        on_ranks(peer, local);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "tensor-parallel Program failed in a noexcept transition: %s\n",
                     error.what());
        std::abort();
    } catch (...) {
        std::fprintf(stderr, "tensor-parallel Program failed in a noexcept transition\n");
        std::abort();
    }
}

SequenceHandle Program::peer_sequence(const SequenceHandle& handle) const noexcept {
    return Access::make_sequence(peer_.get(), Access::lane(handle), Access::epoch(handle));
}

std::vector<SequenceHandle> Program::peer_sequences(std::span<const SequenceHandle> handles) const {
    std::vector<SequenceHandle> out;
    out.reserve(handles.size());
    for (const auto& handle : handles) { out.push_back(peer_sequence(handle)); }
    return out;
}

CaptureOffer Program::peer_offer(const CaptureOffer& offer) const noexcept {
    return Access::make_capture_offer(peer_.get(), Access::lane(offer), Access::epoch(offer),
                                      Access::id(offer));
}

SharedPrefixHandle Program::peer_shared(const SharedPrefixHandle& handle) const noexcept {
    return Access::make_shared_prefix(peer_.get(), Access::index(handle), Access::epoch(handle));
}

runtime::CancellationFlagView
Program::snapshot_cancellation(runtime::CancellationFlagView cancellation) noexcept {
    // Both ranks must observe one cancellation decision per call.
    if (!peer_) { return cancellation; }
    cancellation_snapshot_.store(cancellation.requested(), std::memory_order_release);
    return runtime::CancellationFlagView{&cancellation_snapshot_};
}

RequestBasePlan Program::plan_request(const PreparedPrompt& prompt,
                                      const runtime::ResolvedExecutionOptions& options) {
    return impl_->plan_request(PreparedPromptAccess::view(prompt), options);
}

std::vector<float> Program::causal_score(PreparedPrompt&& prompt, std::uint32_t first_target) {
    PreparedPromptData data = PreparedPromptAccess::take(std::move(prompt));
    if (!peer_) { return impl_->causal_score(std::move(data), first_target); }
    PreparedPromptData peer_data = data.tensor_parallel_peer_copy();
    std::vector<float> local, peer;
    on_ranks([&] { peer = peer_->causal_score(std::move(peer_data), first_target); },
             [&] { local = impl_->causal_score(std::move(data), first_target); });
    require_same(local == peer, "causal scores");
    return local;
}

std::optional<AdmissionCandidate> Program::inspect_admission(
    const PreparedPrompt& prompt, const RequestBasePlan& base, runtime::LaneId destination,
    const ContinuationHandle* source, const SharedPrefixHandle* shared_source,
    std::optional<runtime::CheckpointRef> checkpoint, bool must_retain_private_source) {
    return impl_->inspect_admission(PreparedPromptAccess::view(prompt), base, destination, source,
                                    shared_source, checkpoint, must_retain_private_source);
}

std::optional<ResourcePlan> Program::seal_identity(const AdmissionCandidate& admission,
                                                   const PreparedPrompt& prompt,
                                                   runtime::FinalScheduleIntent intent) {
    std::optional<AdmissionCandidate> sealed = impl_->seal_materialization(
        admission, PreparedPromptAccess::view(prompt), {}, {}, {}, {}, {}, {});
    if (!sealed) { return std::nullopt; }
    impl_->select_shared_captures(*sealed, PreparedPromptAccess::view(prompt),
                                  intent.shared_capture_frontiers);
    if (impl_->revalidate_materialization(*sealed, PreparedPromptAccess::view(prompt)) !=
        runtime::PreflightStatus::Ready) {
        return std::nullopt;
    }
    const bool needs_transfer = sealed->impl_->needs_transfer;
    return ResourcePlan(std::move(*sealed), impl_->resource_revision(), needs_transfer);
}

PressurePlanningSession
Program::begin_pressure_planning(std::span<const AdmissionCandidate* const> candidates,
                                 std::span<const runtime::PlanningCandidateId> candidate_ids,
                                 std::span<const ContinuationHandle* const> private_owners,
                                 std::span<const runtime::PlanningOwnerId> private_owner_ids,
                                 std::span<const SharedPrefixHandle* const> shared_owners,
                                 std::span<const runtime::PlanningOwnerId> shared_owner_ids,
                                 std::span<const runtime::PlanningOwnerId> recency_order) {
    using SessionImpl = detail::PressurePlanningSessionImpl;
    std::vector<SessionImpl::PhysicalCandidateBinding> physical_candidates;
    physical_candidates.reserve(candidates.size());
    for (const AdmissionCandidate* candidate : candidates) {
        if (candidate == nullptr || candidate->impl_ == nullptr) {
            throw std::invalid_argument("pressure planning candidate is empty");
        }
        physical_candidates.push_back(SessionImpl::PhysicalCandidateBinding{
            .state     = candidate->impl_.get(),
            .admission = candidate->impl_.get(),
        });
    }
    return PressurePlanningSession(std::make_unique<detail::PressurePlanningSessionImpl>(
        *impl_, physical_candidates, candidate_ids, private_owners, private_owner_ids,
        shared_owners, shared_owner_ids, recency_order));
}

runtime::PrefillWork
Program::shared_capture_split_prefill_work(const AdmissionCandidate& candidate,
                                           const PreparedPrompt& prompt,
                                           std::span<const std::uint32_t> frontiers) {
    if (impl_ == nullptr) { throw std::logic_error("Program is empty"); }
    return impl_->shared_capture_split_prefill_work(candidate, PreparedPromptAccess::view(prompt),
                                                    frontiers);
}

runtime::ContextTransactionReserveStatus
Program::start_resource_transaction(ResourcePlan&& plan, PreparedPrompt&& prompt,
                                    runtime::CancellationFlagView cancellation) {
    if (plan.revision_.value == 0 || plan.revision_ != impl_->resource_revision()) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    PreparedPromptData data = PreparedPromptAccess::take(std::move(prompt));
    if (!peer_) {
        return impl_->reserve_materialization(std::move(plan.admission_), std::move(data),
                                              cancellation);
    }
    require_same(plan.revision_ == peer_->resource_revision(), "resource revision");
    AdmissionCandidate peer_admission(
        std::make_unique<detail::AdmissionCandidateImpl>(*plan.admission_.impl_));
    PreparedPromptData peer_data = data.tensor_parallel_peer_copy();
    const auto fixed             = snapshot_cancellation(cancellation);
    runtime::ContextTransactionReserveStatus local{}, peer{};
    on_ranks(
        [&] {
            peer = peer_->reserve_materialization(std::move(peer_admission), std::move(peer_data),
                                                  fixed);
        },
        [&] {
            local = impl_->reserve_materialization(std::move(plan.admission_), std::move(data),
                                                   fixed);
        });
    require_same(local == peer, "materialization reservation");
    return local;
}

std::optional<PersistentBackfillProof>
Program::prove_persistent_backfill(const RequestBasePlan& blocked_head,
                                   const ResourcePlan& candidate,
                                   std::span<const SequenceHandle> persistent_borrowers) const {
    if (candidate.revision_.value == 0 || candidate.revision_ != impl_->resource_revision() ||
        !impl_->persistent_backfill_safe(blocked_head, candidate.admission_,
                                         persistent_borrowers)) {
        return std::nullopt;
    }
    return PersistentBackfillProof(candidate.revision_);
}

ContextTransactionProgress
Program::progress_context_transaction(runtime::CancellationFlagView cancellation) {
    if (!peer_) { return impl_->progress_context_transaction(cancellation); }
    const auto fixed = snapshot_cancellation(cancellation);
    std::optional<ContextTransactionProgress> local, peer;
    on_ranks([&] { peer.emplace(peer_->progress_context_transaction(fixed)); },
             [&] { local.emplace(impl_->progress_context_transaction(fixed)); });
    require_same(local->index() == peer->index(), "context transaction progress");
    if (const auto* result = std::get_if<MaterializationResult>(&*local)) {
        require_same(result->status == std::get<MaterializationResult>(*peer).status,
                     "materialization result");
    } else if (const auto* result = std::get_if<ActiveCaptureResult>(&*local)) {
        require_same(result->status == std::get<ActiveCaptureResult>(*peer).status,
                     "active capture result");
    }
    return std::move(*local);
}

void Program::finalize_context_transaction() noexcept {
    on_ranks_noexcept([&] { peer_->finalize_context_transaction(); },
                      [&] { impl_->finalize_context_transaction(); });
}

bool Program::has_context_transaction() const noexcept { return impl_->has_context_transaction(); }

bool Program::wait_context_transfer() noexcept {
    if (!peer_) { return impl_->wait_context_transfer(); }
    bool local = false, peer = false;
    on_ranks_noexcept([&] { peer = peer_->wait_context_transfer(); },
                      [&] { local = impl_->wait_context_transfer(); });
    return local || peer;
}

PrefillProgress Program::advance_prefill(SequenceHandle sequence,
                                         runtime::ExecutionTiming* failed_timing) {
    if (!peer_) { return impl_->advance_prefill(sequence, failed_timing); }
    const SequenceHandle peer_handle = peer_sequence(sequence);
    runtime::ExecutionTiming peer_failed;
    std::optional<PrefillProgress> local, peer;
    on_ranks([&] {
                 peer.emplace(peer_->advance_prefill(peer_handle,
                                                     failed_timing ? &peer_failed : nullptr));
             },
             [&] { local.emplace(impl_->advance_prefill(sequence, failed_timing)); });
    require_same(local->complete == peer->complete &&
                     local->processed_prompt_tokens == peer->processed_prompt_tokens &&
                     local->pending.has_value() == peer->pending.has_value() &&
                     local->capture.has_value() == peer->capture.has_value(),
                 "prefill progress");
    if (local->pending) {
        require_same(same_tokens(*local->pending, *peer->pending), "prefill tokens");
        peer_pending_.reset();
        peer_pending_.emplace(std::move(*peer->pending));
    }
    return std::move(*local);
}

CaptureAssessment
Program::inspect_capture(const CaptureOffer& offer, const SharedPrefixHandle* exact_shared,
                         const SharedPrefixHandle* replacement,
                         std::optional<runtime::CheckpointRef> private_replacement,
                         bool permit_shared_publication) const {
    return impl_->inspect_capture(offer, exact_shared, replacement, private_replacement,
                                  permit_shared_publication);
}

std::vector<runtime::CheckpointRecoveryAlternativeWork>
Program::checkpoint_recovery_work(const ContinuationHandle& owner,
                                  runtime::CheckpointRef checkpoint) const {
    return impl_->checkpoint_recovery_work(owner, checkpoint);
}

CapturePressurePlanningSession Program::begin_capture_pressure_planning(
    const CaptureAssessment& assessment, std::span<const ContinuationHandle* const> private_owners,
    std::span<const runtime::PlanningOwnerId> private_owner_ids,
    std::span<const SharedPrefixHandle* const> shared_owners,
    std::span<const runtime::PlanningOwnerId> shared_owner_ids,
    std::span<const runtime::PlanningOwnerId> recency_order) {
    CapturePressureCandidate candidate(impl_->make_capture_physical_candidate(assessment));
    using SessionImpl = detail::PressurePlanningSessionImpl;
    const std::array physical_candidates{SessionImpl::PhysicalCandidateBinding{
        .state   = candidate.impl_.get(),
        .capture = candidate.impl_.get(),
    }};
    const std::array candidate_ids{CapturePressurePlanningSession::candidate_id()};
    PressurePlanningSession session(std::make_unique<detail::PressurePlanningSessionImpl>(
        *impl_, physical_candidates, candidate_ids, private_owners, private_owner_ids,
        shared_owners, shared_owner_ids, recency_order));
    return CapturePressurePlanningSession(std::move(candidate), std::move(session));
}

std::vector<runtime::CheckpointRecoveryAlternativeWork>
Program::checkpoint_recovery_work(const SharedPrefixHandle& owner,
                                  runtime::CheckpointRef checkpoint) const {
    return impl_->checkpoint_recovery_work(owner, checkpoint);
}

bool Program::shared_capture_matches(const CaptureOffer& offer,
                                     const SharedPrefixHandle& shared) const {
    return impl_->shared_capture_matches(offer, shared);
}

void Program::skip_capture(CaptureOffer&& offer) {
    if (!peer_) {
        impl_->skip_capture(std::move(offer));
        return;
    }
    CaptureOffer peer_copy = peer_offer(offer);
    on_ranks([&] { peer_->skip_capture(std::move(peer_copy)); },
             [&] { impl_->skip_capture(std::move(offer)); });
}

runtime::ContextTransactionReserveStatus
Program::reserve_active_capture(CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
                                const SharedPrefixHandle* replacement,
                                std::optional<runtime::CheckpointRef> private_replacement,
                                bool permit_shared_publication,
                                runtime::CancellationFlagView cancellation) {
    if (!peer_) {
        return impl_->reserve_active_capture(std::move(offer), exact_shared, replacement,
                                             private_replacement, permit_shared_publication,
                                             cancellation);
    }
    CaptureOffer peer_copy = peer_offer(offer);
    std::optional<SharedPrefixHandle> peer_exact, peer_replacement;
    if (exact_shared) { peer_exact.emplace(peer_shared(*exact_shared)); }
    if (replacement) { peer_replacement.emplace(peer_shared(*replacement)); }
    const auto fixed = snapshot_cancellation(cancellation);
    runtime::ContextTransactionReserveStatus local{}, peer{};
    on_ranks(
        [&] {
            peer = peer_->reserve_active_capture(
                std::move(peer_copy), peer_exact ? &*peer_exact : nullptr,
                peer_replacement ? &*peer_replacement : nullptr, private_replacement,
                permit_shared_publication, fixed);
        },
        [&] {
            local = impl_->reserve_active_capture(std::move(offer), exact_shared, replacement,
                                                  private_replacement, permit_shared_publication,
                                                  fixed);
        });
    require_same(local == peer, "active capture reservation");
    return local;
}

runtime::ContextTransactionReserveStatus Program::reserve_active_capture_with_pressure(
    CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
    const SharedPrefixHandle* replacement,
    std::optional<runtime::CheckpointRef> private_replacement, bool permit_shared_publication,
    CapturePressurePlan&& pressure, runtime::CancellationFlagView cancellation) {
    if (pressure.revision_.value == 0 || pressure.revision_ != impl_->resource_revision()) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    if (!peer_) {
        return impl_->reserve_active_capture_with_pressure(
            std::move(offer), exact_shared, replacement, private_replacement,
            permit_shared_publication, std::move(pressure.pressure_), cancellation);
    }
    require_same(pressure.revision_ == peer_->resource_revision(), "resource revision");
    CaptureOffer peer_copy = peer_offer(offer);
    std::optional<SharedPrefixHandle> peer_exact, peer_replacement;
    if (exact_shared) { peer_exact.emplace(peer_shared(*exact_shared)); }
    if (replacement) { peer_replacement.emplace(peer_shared(*replacement)); }
    CapturePressureCandidate peer_pressure(
        std::make_unique<detail::CapturePressureCandidateImpl>(*pressure.pressure_.impl_));
    const auto fixed = snapshot_cancellation(cancellation);
    runtime::ContextTransactionReserveStatus local{}, peer{};
    on_ranks(
        [&] {
            peer = peer_->reserve_active_capture_with_pressure(
                std::move(peer_copy), peer_exact ? &*peer_exact : nullptr,
                peer_replacement ? &*peer_replacement : nullptr, private_replacement,
                permit_shared_publication, std::move(peer_pressure), fixed);
        },
        [&] {
            local = impl_->reserve_active_capture_with_pressure(
                std::move(offer), exact_shared, replacement, private_replacement,
                permit_shared_publication, std::move(pressure.pressure_), fixed);
        });
    require_same(local == peer, "pressured active capture reservation");
    return local;
}

PendingBatch Program::decode(std::span<const SequenceHandle> sequences,
                             std::span<const runtime::RoundBudget> budgets,
                             runtime::ExecutionTiming* failed_timing) {
    if (!peer_) { return impl_->decode(sequences, budgets, failed_timing); }
    const auto peer_handles = peer_sequences(sequences);
    runtime::ExecutionTiming peer_failed;
    std::optional<PendingBatch> local, peer;
    on_ranks([&] {
                 peer.emplace(peer_->decode(peer_handles, budgets,
                                            failed_timing ? &peer_failed : nullptr));
             },
             [&] { local.emplace(impl_->decode(sequences, budgets, failed_timing)); });
    require_same(same_tokens(*local, *peer), "decode tokens");
    peer_pending_.reset();
    peer_pending_.emplace(std::move(*peer));
    return std::move(*local);
}

runtime::ExecutionTiming
Program::append_forced_tokens(std::span<const SequenceHandle> sequences,
                              std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                              std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
                              runtime::ExecutionTiming* failed_timing) {
    if (!peer_) {
        return impl_->append_forced_tokens(sequences, row_major_tokens, row_stride,
                                           prefix_execution_splits, failed_timing);
    }
    const auto peer_handles = peer_sequences(sequences);
    runtime::ExecutionTiming local{}, peer_failed{};
    on_ranks(
        [&] {
            (void)peer_->append_forced_tokens(peer_handles, row_major_tokens, row_stride,
                                              prefix_execution_splits,
                                              failed_timing ? &peer_failed : nullptr);
        },
        [&] {
            local = impl_->append_forced_tokens(sequences, row_major_tokens, row_stride,
                                                prefix_execution_splits, failed_timing);
        });
    return local;
}

CommitResult Program::commit(PendingBatch&& pending,
                             std::span<const runtime::CommitDecision> decisions,
                             runtime::CommitObservation observation,
                             runtime::ExecutionTiming* failed_timing) {
    if (!peer_) { return impl_->commit(std::move(pending), decisions, observation, failed_timing); }
    if (!peer_pending_) { ranks_diverged("commit without a peer pending batch"); }
    PendingBatch peer_batch(std::move(*peer_pending_));
    peer_pending_.reset();
    runtime::ExecutionTiming peer_failed;
    std::optional<CommitResult> local, peer;
    on_ranks(
        [&] {
            peer.emplace(peer_->commit(std::move(peer_batch), decisions, observation,
                                       failed_timing ? &peer_failed : nullptr));
        },
        [&] { local.emplace(impl_->commit(std::move(pending), decisions, observation,
                                          failed_timing)); });
    bool same = local->row_count == peer->row_count;
    for (std::size_t row = 0; same && row < local->row_count; ++row) {
        same = local->rows[row].disposition == peer->rows[row].disposition &&
               local->captures[row].has_value() == peer->captures[row].has_value();
    }
    require_same(same, "commit result");
    return std::move(*local);
}

DiscardResult Program::abort_pending(PendingBatch&& pending) noexcept {
    if (!peer_) { return impl_->abort_pending(std::move(pending)); }
    DiscardResult local{};
    std::optional<PendingBatch> peer_batch;
    if (peer_pending_) {
        peer_batch.emplace(std::move(*peer_pending_));
        peer_pending_.reset();
    }
    on_ranks_noexcept(
        [&] {
            if (peer_batch) { (void)peer_->abort_pending(std::move(*peer_batch)); }
        },
        [&] { local = impl_->abort_pending(std::move(pending)); });
    return local;
}

FinishResult Program::finish(SequenceHandle sequence) noexcept {
    if (!peer_) { return impl_->finish(sequence); }
    const SequenceHandle peer_handle = peer_sequence(sequence);
    std::optional<FinishResult> local;
    on_ranks_noexcept([&] { (void)peer_->finish(peer_handle); },
                      [&] { local.emplace(impl_->finish(sequence)); });
    return std::move(*local);
}

AbortResult Program::abort(SequenceHandle sequence) noexcept {
    if (!peer_) { return impl_->abort(sequence); }
    const SequenceHandle peer_handle = peer_sequence(sequence);
    std::optional<AbortResult> local;
    on_ranks_noexcept([&] { (void)peer_->abort(peer_handle); },
                      [&] { local.emplace(impl_->abort(sequence)); });
    return std::move(*local);
}

std::optional<std::uint32_t> Program::device_kv_lease_settlement_tokens(
    SequenceHandle sequence, std::uint32_t forced_span_tokens) const noexcept {
    return impl_->device_kv_lease_settlement_tokens(sequence, forced_span_tokens);
}

std::optional<DeviceKVLeaseShortfall>
Program::device_kv_lease_shortfall(SequenceHandle sequence) const noexcept {
    return impl_->device_kv_lease_shortfall(sequence);
}

bool Program::resume_device_kv_lease(SequenceHandle sequence) noexcept {
    if (!peer_) { return impl_->resume_device_kv_lease(sequence); }
    const SequenceHandle peer_handle = peer_sequence(sequence);
    bool local = false, peer = false;
    on_ranks_noexcept([&] { peer = peer_->resume_device_kv_lease(peer_handle); },
                      [&] { local = impl_->resume_device_kv_lease(sequence); });
    if (local != peer) { ranks_diverged("device KV lease resumption"); }
    return local;
}

DeviceKVPages
Program::retained_device_kv_pages(const ContinuationHandle& continuation) const noexcept {
    return impl_->retained_device_kv_pages(continuation);
}

DeviceKVPages Program::retained_device_kv_pages(const SharedPrefixHandle& shared) const noexcept {
    return impl_->retained_device_kv_pages(shared);
}

ReleaseResult Program::release_continuation(ContinuationHandle&& continuation) noexcept {
    if (!peer_) { return impl_->release_continuation(std::move(continuation)); }
    ContinuationHandle peer_handle = Access::make_continuation(
        peer_.get(), Access::index(continuation), Access::epoch(continuation));
    ReleaseResult local{};
    on_ranks_noexcept([&] { (void)peer_->release_continuation(std::move(peer_handle)); },
                      [&] { local = impl_->release_continuation(std::move(continuation)); });
    return local;
}

ReleaseResult Program::release_shared_prefix(SharedPrefixHandle&& shared) noexcept {
    if (!peer_) { return impl_->release_shared_prefix(std::move(shared)); }
    SharedPrefixHandle peer_handle = peer_shared(shared);
    ReleaseResult local{};
    on_ranks_noexcept([&] { (void)peer_->release_shared_prefix(std::move(peer_handle)); },
                      [&] { local = impl_->release_shared_prefix(std::move(shared)); });
    return local;
}

std::optional<PhysicalUsageSnapshot> Program::fail_all_cleanup() noexcept {
    return cleanup_ranks(detail::ProgramCleanup::Failure);
}

std::optional<PhysicalUsageSnapshot> Program::shutdown_cleanup() noexcept {
    return cleanup_ranks(detail::ProgramCleanup::Shutdown);
}

std::optional<PhysicalUsageSnapshot>
Program::cleanup_ranks(detail::ProgramCleanup cleanup) noexcept {
    if (!peer_) { return impl_->fail_all_cleanup(cleanup); }
    peer_pending_.reset();
    std::optional<PhysicalUsageSnapshot> usage;
    on_ranks_noexcept([&] { (void)peer_->fail_all_cleanup(cleanup); },
                      [&] { usage = impl_->fail_all_cleanup(cleanup); });
    return usage;
}

bool Program::isolated_request_feasible(const RequestBasePlan& base) const noexcept {
    return impl_->isolated_request_feasible(base);
}

bool Program::hybrid_prefix_cache() const noexcept { return impl_->hybrid_prefix_cache(); }

HybridAdmissionQuote Program::hybrid_quote(const PreparedPrompt& prompt,
                                           const RequestBasePlan& base,
                                           runtime::LaneId destination) {
    return impl_->hybrid_quote(PreparedPromptAccess::view(prompt), base, destination);
}

runtime::ContextTransactionReserveStatus
Program::hybrid_reserve_materialization(HybridAdmissionQuote&& quote, PreparedPrompt&& prompt,
                                        runtime::CancellationFlagView cancellation) {
    // The prompt is taken only once the reservation will succeed: a rejected quote leaves the
    // waiting request intact for a later admission attempt.
    if (!impl_->hybrid_reservable(quote, cancellation)) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    return impl_->hybrid_reserve_materialization(
        std::move(quote), PreparedPromptAccess::view(prompt),
        [&prompt]() { return PreparedPromptAccess::take(std::move(prompt)); }, cancellation);
}

std::uint32_t Program::hybrid_reclaim_device_kv(std::uint32_t main_pages,
                                                std::uint32_t backend_pages) {
    return impl_->hybrid_reclaim_device_kv(main_pages, backend_pages);
}

std::optional<std::uint32_t> Program::hybrid_prefetch(const PreparedPrompt& prompt,
                                                      const RequestBasePlan& base) {
    return impl_->hybrid_prefetch(PreparedPromptAccess::view(prompt), base);
}

std::uint32_t Program::hybrid_prefetch_room() const noexcept {
    return impl_->hybrid_prefetch_room();
}

HybridPrefixCacheStats Program::hybrid_stats() const noexcept { return impl_->hybrid_stats(); }

void Program::set_hybrid_cost(const runtime::prefix_cache::CacheCostModel& cost) {
    impl_->set_hybrid_cost(cost);
}

void Program::set_hybrid_coalesce_wait_limit(double seconds) {
    impl_->set_hybrid_coalesce_wait_limit(seconds);
}

HybridCachePersistence Program::attach_hybrid_cache_file(const std::filesystem::path& path,
                                                         std::string fingerprint,
                                                         const StartupObserver& observer) {
    return impl_->attach_hybrid_cache_file(path, std::move(fingerprint), observer);
}

std::optional<HybridCachePersistence> Program::hybrid_shutdown_save() const {
    return impl_->hybrid_shutdown_save();
}

runtime::ProgramResourceRevision Program::resource_revision() const noexcept {
    return impl_->resource_revision();
}

PhysicalUsageSnapshot Program::physical_usage() const noexcept { return impl_->physical_usage(); }

MemorySummary Program::memory_summary() const noexcept { return impl_->memory_summary(); }

void Program::reset_memory_peaks() noexcept {
    if (!peer_) {
        impl_->reset_memory_peaks();
        return;
    }
    on_ranks_noexcept([&] { peer_->reset_memory_peaks(); }, [&] { impl_->reset_memory_peaks(); });
}

SequencePlanner make_sequence_planner(const execution::Parameters& parameters,
                                      DeviceContext& device, const EngineOptions& options) {
    return SequencePlanner(detail::make_sequence_planner_impl(parameters, device, options));
}

std::unique_ptr<Program> create_program(const execution::Parameters& parameters,
                                        SequencePlan&& plan, DeviceContext& device,
                                        const StartupObserver& startup_observer) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("sequence plan is empty"); }
    if (plan.impl_->parameters != &parameters) {
        throw std::invalid_argument("sequence plan belongs to another model instance");
    }
    if (parameters.model.config().tensor_parallel.split()) {
        throw std::invalid_argument("a tensor-parallel shard needs create_tensor_parallel_program");
    }
    auto impl =
        std::make_unique<detail::ProgramImpl>(parameters, *plan.impl_, device, startup_observer);
    plan.impl_.reset();
    return std::unique_ptr<Program>(new Program(std::move(impl)));
}

std::unique_ptr<Program>
create_tensor_parallel_program(std::array<const execution::Parameters*, 2> parameters,
                               std::array<SequencePlan, 2>&& plans,
                               std::array<DeviceContext*, 2> devices,
                               std::array<const TensorParallelDeviceView*, 2> links,
                               const StartupObserver& startup_observer) {
    for (std::size_t rank = 0; rank < 2; ++rank) {
        if (parameters[rank] == nullptr || devices[rank] == nullptr || links[rank] == nullptr) {
            throw std::invalid_argument("tensor-parallel Program rank is incomplete");
        }
        const auto placement = parameters[rank]->model.config().tensor_parallel;
        if (placement.size != 2 || placement.rank != rank) {
            throw std::invalid_argument("tensor-parallel Program rank has another placement");
        }
        if (plans[rank].impl_ == nullptr || plans[rank].impl_->parameters != parameters[rank]) {
            throw std::invalid_argument("tensor-parallel sequence plan belongs to another rank");
        }
        if (links[rank]->rank != static_cast<std::int32_t>(rank)) {
            throw std::invalid_argument("tensor-parallel link view belongs to another rank");
        }
    }
    if (plans[0].impl_->kv_capacity != plans[1].impl_->kv_capacity ||
        plans[0].impl_->capacity != plans[1].impl_->capacity) {
        throw std::invalid_argument("tensor-parallel ranks were planned with different capacity");
    }
    auto executor = std::make_unique<detail::PeerExecutor>(*devices[1]);
    const StartupObserver quiet;
    std::unique_ptr<detail::ProgramImpl> local, peer;
    executor->run(
        [&] {
            peer = std::make_unique<detail::ProgramImpl>(*parameters[1], *plans[1].impl_,
                                                         *devices[1], quiet, links[1]);
        },
        [&] {
            local = std::make_unique<detail::ProgramImpl>(*parameters[0], *plans[0].impl_,
                                                          *devices[0], startup_observer, links[0]);
        });
    plans[0].impl_.reset();
    plans[1].impl_.reset();
    return std::unique_ptr<Program>(
        new Program(std::move(local), std::move(peer), std::move(executor)));
}

} // namespace ninfer::models::qwen3_5

namespace ninfer::models::qwen3_5 {

CaptureAssessment::CaptureAssessment()
    : implementation(std::make_shared<detail::CaptureAssessmentImpl>()) {}

AdmissionCandidate::AdmissionCandidate(
    std::unique_ptr<detail::AdmissionCandidateImpl> impl) noexcept
    : impl_(std::move(impl)) {}

AdmissionCandidate::AdmissionCandidate(AdmissionCandidate&&) noexcept = default;

AdmissionCandidate& AdmissionCandidate::operator=(AdmissionCandidate&&) noexcept = default;

AdmissionCandidate::~AdmissionCandidate() = default;

CapturePressureCandidate::CapturePressureCandidate(
    std::unique_ptr<detail::CapturePressureCandidateImpl> impl) noexcept
    : impl_(std::move(impl)) {}

CapturePressureCandidate::CapturePressureCandidate(CapturePressureCandidate&&) noexcept = default;

CapturePressureCandidate&
CapturePressureCandidate::operator=(CapturePressureCandidate&&) noexcept = default;

CapturePressureCandidate::~CapturePressureCandidate() = default;

const runtime::RequestPlanSummary& AdmissionCandidate::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const runtime::IdentityMaterializationAssessment&
AdmissionCandidate::identity_assessment() const noexcept {
    static const runtime::IdentityMaterializationAssessment empty;
    return impl_ != nullptr ? impl_->identity_assessment : empty;
}

} // namespace ninfer::models::qwen3_5
