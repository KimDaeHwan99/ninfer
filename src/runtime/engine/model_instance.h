#pragma once

#include "core/device.h"
#include "core/tensor_parallel.h"
#include "models/qwen3_5/model.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/runtime_types.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/kv_capacity.h"

#include <array>
#include <memory>

namespace ninfer::runtime {

[[nodiscard]] EngineOptions normalize_engine_options(EngineOptions options);

// The second rank of a tensor-parallel instance: its device, shard Model and Parameters, and the
// link both ranks' collectives use. Owned by ModelInstance so it outlives the Program.
struct TensorParallelPeer {
    DeviceContext device;
    std::unique_ptr<models::qwen3_5::Model> model;
    std::unique_ptr<const models::qwen3_5::execution::Parameters> parameters;
    std::unique_ptr<TensorParallelLink> link;
    std::array<TensorParallelDeviceView, 2> views{};

    explicit TensorParallelPeer(int device_id) : device(device_id) {}
};

struct ModelInstance {
    using ModelContract = models::qwen3_5::RuntimeTypes;

    std::unique_ptr<models::qwen3_5::Model> model;
    const models::qwen3_5::execution::Parameters parameters;
    models::qwen3_5::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<TensorParallelPeer> peer;
    std::unique_ptr<models::qwen3_5::Program> program;

    ModelInstance(std::unique_ptr<models::qwen3_5::Model> model, const EngineOptions& options);
    ~ModelInstance();
    ModelInstance(const ModelInstance&)            = delete;
    ModelInstance& operator=(const ModelInstance&) = delete;
};

struct ConstructedModel {
    std::unique_ptr<ModelInstance> instance;
    LoadSummary load;
    ModelMetadata model_metadata;
    ContextMachineCostModel context_cost;
    // The options this instance was built from, with any value the model had to resolve (today the
    // single host RAM budget's Host split and long-anchor count) replaced by what the plan actually
    // uses. Carrying it keeps the Engine's copy, its ResourceManager and the frontend grid on one
    // resolved value instead of a plan that silently differs from the reported options.
    EngineOptions options;
};

[[nodiscard]] ConstructedModel construct_model(const EngineOptions& options, DeviceContext& device);

} // namespace ninfer::runtime
