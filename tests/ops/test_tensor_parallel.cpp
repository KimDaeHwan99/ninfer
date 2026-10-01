// Two-device qualification of the tensor-parallel collectives: FP64 oracle for the residual
// all-reduce, exact comparison for the row gather, bit-identity of the two ranks, back-to-back
// chains without host synchronization, and independent per-rank CUDA Graph capture.
#include "ninfer/ops/tensor_parallel.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::size_t kSlotBytes = 24ULL * 1024 * 1024;

struct RankBuffers {
    void* partial     = nullptr;
    void* residual    = nullptr;
    void* local       = nullptr;
    void* destination = nullptr;
    cudaStream_t stream = nullptr;
};

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) out[i] = f32_to_bf16(values[i]);
    return out;
}

// Runs `body(rank)` on two threads bound to the two ranks' devices.
void on_ranks(const std::array<int, 2>& devices, const std::function<void(int)>& body) {
    std::thread peer([&] {
        check(cudaSetDevice(devices[1]), "bind rank 1");
        body(1);
    });
    check(cudaSetDevice(devices[0]), "bind rank 0");
    body(0);
    peer.join();
}

int residual_case(TensorParallelLink& link, std::array<RankBuffers, 2>& ranks,
                  const std::array<int, 2>& devices, std::int32_t rows, std::int32_t columns,
                  int chain, std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::array<std::vector<std::vector<float>>, 2> partials;
    std::vector<float> residual(count);
    fill_uniform(residual, seed, -4.0f, 4.0f);
    round_to_bf16(residual);
    for (int r = 0; r < 2; ++r) {
        partials[r].resize(chain);
        for (int step = 0; step < chain; ++step) {
            partials[r][step].resize(count);
            fill_uniform(partials[r][step], seed + 17 * (step + 1) + 1000 * r, -2.0f, 2.0f);
            round_to_bf16(partials[r][step]);
        }
    }
    // Oracle: the same chain in FP64, each step rounded once to BF16 storage like the device.
    std::vector<double> expected(residual.begin(), residual.end());
    std::vector<float> stored = residual;
    for (int step = 0; step < chain; ++step) {
        for (std::size_t i = 0; i < count; ++i) {
            const double value = static_cast<double>(stored[i]) + partials[0][step][i] +
                                 partials[1][step][i];
            expected[i] = value;
            stored[i]   = bf16_to_f32(f32_to_bf16(static_cast<float>(value)));
        }
    }
    std::array<std::vector<std::uint16_t>, 2> results;
    on_ranks(devices, [&](int r) {
        auto& b                 = ranks[r];
        const auto residual_bits = bf16_bits(residual);
        check(cudaMemcpy(b.residual, residual_bits.data(), count * 2, cudaMemcpyHostToDevice),
              "upload residual");
        // Stage every partial of the chain first so no host synchronization separates the steps.
        std::vector<void*> staged(chain);
        for (int step = 0; step < chain; ++step) {
            check(cudaMalloc(&staged[step], count * 2), "partial chain");
            const auto bits = bf16_bits(partials[r][step]);
            check(cudaMemcpy(staged[step], bits.data(), count * 2, cudaMemcpyHostToDevice),
                  "upload partial");
        }
        check(cudaDeviceSynchronize(), "uploads");
        const auto view = link.view(r);
        for (int step = 0; step < chain; ++step) {
            Tensor p(staged[step], DType::BF16, {rows, columns});
            Tensor x(b.residual, DType::BF16, {rows, columns});
            ops::tp_residual_allreduce(p, x, view, b.stream);
        }
        check(cudaStreamSynchronize(b.stream), "chain");
        results[r].resize(count);
        check(cudaMemcpy(results[r].data(), b.residual, count * 2, cudaMemcpyDeviceToHost),
              "download residual");
        for (void* p : staged) check(cudaFree(p), "free partial");
    });
    char label[128];
    std::snprintf(label, sizeof(label), "tp_residual_allreduce [%d,%d] x%d", rows, columns, chain);
    std::vector<double> device(count);
    for (std::size_t i = 0; i < count; ++i) device[i] = bf16_to_f32(results[0][i]);
    // The final step's arithmetic is checked against the FP64 value of that step; earlier steps
    // are fed back through BF16 storage exactly as the oracle does.
    int failures = verify_pointwise(label, device, expected, {0.0, 3.95e-3});
    failures += verify_exact("tp_residual_allreduce rank identity", results[1], results[0]);
    return failures;
}

int gather_case(TensorParallelLink& link, std::array<RankBuffers, 2>& ranks,
                const std::array<int, 2>& devices, std::int32_t rows, std::int32_t columns,
                std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::array<std::vector<std::uint16_t>, 2> local;
    for (int r = 0; r < 2; ++r) {
        std::vector<float> values(count);
        fill_uniform(values, seed + r, -30.0f, 30.0f);
        local[r] = bf16_bits(values);
    }
    std::vector<std::uint16_t> expected(2 * count);
    for (std::int32_t c = 0; c < columns; ++c) {
        for (int r = 0; r < 2; ++r) {
            for (std::int32_t i = 0; i < rows; ++i) {
                expected[static_cast<std::size_t>(c) * 2 * rows + r * rows + i] =
                    local[r][static_cast<std::size_t>(c) * rows + i];
            }
        }
    }
    std::array<std::vector<std::uint16_t>, 2> results;
    on_ranks(devices, [&](int r) {
        auto& b = ranks[r];
        check(cudaMemcpy(b.local, local[r].data(), count * 2, cudaMemcpyHostToDevice), "local");
        check(cudaMemsetAsync(b.destination, 0xff, 2 * count * 2, b.stream), "poison");
        // Pageable uploads complete on the legacy stream, which the non-blocking rank stream
        // does not order against.
        check(cudaDeviceSynchronize(), "uploads");
        Tensor l(b.local, DType::BF16, {rows, columns});
        Tensor d(b.destination, DType::BF16, {2 * rows, columns});
        ops::tp_allgather_rows(l, d, link.view(r), b.stream);
        check(cudaStreamSynchronize(b.stream), "gather");
        results[r].resize(2 * count);
        check(cudaMemcpy(results[r].data(), b.destination, 2 * count * 2, cudaMemcpyDeviceToHost),
              "download gather");
    });
    char label[96];
    std::snprintf(label, sizeof(label), "tp_allgather_rows [%d,%d] rank0", rows, columns);
    int failures = verify_exact(label, results[0], expected);
    std::snprintf(label, sizeof(label), "tp_allgather_rows [%d,%d] rank1", rows, columns);
    failures += verify_exact(label, results[1], expected);
    return failures;
}

// Each rank captures its own graph of `steps` decode-sized collectives; replays must keep the
// shared device sequence numbering consistent with eager calls before and after.
int graph_case(TensorParallelLink& link, std::array<RankBuffers, 2>& ranks,
               const std::array<int, 2>& devices) {
    constexpr std::int32_t kRows = 5120;
    constexpr int kSteps         = 16;
    constexpr int kReplays       = 64;
    std::array<double, 2> eager_us{}, graph_us{};
    on_ranks(devices, [&](int r) {
        auto& b         = ranks[r];
        const auto view = link.view(r);
        check(cudaMemsetAsync(b.partial, 0, kRows * 2, b.stream), "zero partial");
        check(cudaMemsetAsync(b.residual, 0, kRows * 2, b.stream), "zero residual");
        Tensor p(b.partial, DType::BF16, {kRows, 1});
        Tensor x(b.residual, DType::BF16, {kRows, 1});
        for (int i = 0; i < 8; ++i) ops::tp_residual_allreduce(p, x, view, b.stream);
        check(cudaStreamSynchronize(b.stream), "warm");
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < kSteps * kReplays; ++i) ops::tp_residual_allreduce(p, x, view, b.stream);
        check(cudaStreamSynchronize(b.stream), "eager");
        eager_us[r] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() -
                                                                start)
                          .count() /
                      (kSteps * kReplays);
        cudaGraph_t graph = nullptr;
        check(cudaStreamBeginCapture(b.stream, cudaStreamCaptureModeThreadLocal), "capture");
        for (int i = 0; i < kSteps; ++i) ops::tp_residual_allreduce(p, x, view, b.stream);
        check(cudaStreamEndCapture(b.stream, &graph), "end capture");
        cudaGraphExec_t exec = nullptr;
        check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
        start = std::chrono::steady_clock::now();
        for (int i = 0; i < kReplays; ++i) check(cudaGraphLaunch(exec, b.stream), "launch");
        check(cudaStreamSynchronize(b.stream), "replay");
        graph_us[r] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() -
                                                                start)
                          .count() /
                      (kSteps * kReplays);
        for (int i = 0; i < 4; ++i) ops::tp_residual_allreduce(p, x, view, b.stream);
        check(cudaStreamSynchronize(b.stream), "eager after graph");
        check(cudaGraphExecDestroy(exec), "destroy exec");
        check(cudaGraphDestroy(graph), "destroy graph");
    });
    std::printf("tp_residual_allreduce [5120,1]: eager %.2f us, graph %.2f us per collective\n",
                eager_us[0], graph_us[0]);
    return 0;
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
        std::printf("SKIP: tensor-parallel test needs two CUDA devices\n");
        return 77;
    }
    const std::array<int, 2> devices{0, 1};
    TensorParallelLink link(devices, kSlotBytes);
    std::array<RankBuffers, 2> ranks;
    on_ranks(devices, [&](int r) {
        auto& b = ranks[r];
        check(cudaStreamCreateWithFlags(&b.stream, cudaStreamNonBlocking), "stream");
        check(cudaMalloc(&b.partial, kSlotBytes), "partial");
        check(cudaMalloc(&b.residual, kSlotBytes), "residual");
        check(cudaMalloc(&b.local, kSlotBytes), "local");
        check(cudaMalloc(&b.destination, 2 * kSlotBytes), "destination");
    });
    int failures = 0;
    failures += residual_case(link, ranks, devices, 5120, 1, 1, 11);
    failures += residual_case(link, ranks, devices, 5120, 4, 6, 12);
    failures += residual_case(link, ranks, devices, 5120, 1, 64, 13);
    failures += residual_case(link, ranks, devices, 5120, 2048, 3, 14);
    failures += residual_case(link, ranks, devices, 8, 3, 5, 15);
    failures += gather_case(link, ranks, devices, 124160, 1, 21);
    failures += gather_case(link, ranks, devices, 124160, 4, 22);
    failures += gather_case(link, ranks, devices, 65536, 48, 23);
    failures += graph_case(link, ranks, devices);
    failures += residual_case(link, ranks, devices, 5120, 3, 9, 16);
    on_ranks(devices, [&](int r) {
        auto& b = ranks[r];
        cudaFree(b.partial);
        cudaFree(b.residual);
        cudaFree(b.local);
        cudaFree(b.destination);
        cudaStreamDestroy(b.stream);
    });
    if (failures) {
        std::printf("FAILED: %d\n", failures);
        return 1;
    }
    std::printf("PASSED\n");
    return 0;
}
