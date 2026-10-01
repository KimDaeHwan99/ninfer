// Two-device qualification of the tensor-parallel collectives: FP64 oracle for the residual
// all-reduce, exact comparison for the row gather, bit-identity of the two ranks, back-to-back
// chains without host synchronization, independent per-rank CUDA Graph capture, and the
// copy-channel asynchronous all-reduce.
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

// Interleaves packed (small) and staged (large) all-reduces with gathers, without host
// synchronization, while rank 1 lags behind rank 0: slot reuse across different payload sizes
// must never expose a payload the peer has not finished reading.
int mixed_chain_case(TensorParallelLink& link, const std::array<int, 2>& devices) {
    constexpr int kSteps = 24;
    const std::array<std::int32_t, 3> columns{1, 64, 4};
    std::array<std::array<std::vector<std::vector<float>>, 3>, 2> partials;
    std::array<std::vector<float>, 3> residual0;
    for (int s = 0; s < 3; ++s) {
        const std::size_t count = std::size_t(5120) * columns[s];
        residual0[s].resize(count);
        fill_uniform(residual0[s], 300 + s, -1.0f, 1.0f);
        round_to_bf16(residual0[s]);
        for (int r = 0; r < 2; ++r) {
            partials[r][s].resize(kSteps);
            for (int k = 0; k < kSteps; ++k) {
                partials[r][s][k].resize(count);
                fill_uniform(partials[r][s][k], 400 + 31 * k + 7 * s + 1000 * r, -0.5f, 0.5f);
                round_to_bf16(partials[r][s][k]);
            }
        }
    }
    std::array<std::array<std::vector<std::uint16_t>, 3>, 2> results;
    on_ranks(devices, [&](int r) {
        cudaStream_t stream = nullptr;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        std::array<void*, 3> residual{};
        std::vector<std::array<void*, 3>> staged(kSteps);
        for (int s = 0; s < 3; ++s) {
            const std::size_t bytes = std::size_t(5120) * columns[s] * 2;
            check(cudaMalloc(&residual[s], bytes), "residual");
            const auto bits = bf16_bits(residual0[s]);
            check(cudaMemcpy(residual[s], bits.data(), bytes, cudaMemcpyHostToDevice), "upload");
            for (int k = 0; k < kSteps; ++k) {
                check(cudaMalloc(&staged[k][s], bytes), "partial");
                const auto pbits = bf16_bits(partials[r][s][k]);
                check(cudaMemcpy(staged[k][s], pbits.data(), bytes, cudaMemcpyHostToDevice),
                      "upload partial");
            }
        }
        check(cudaDeviceSynchronize(), "uploads");
        if (r == 1) { std::this_thread::sleep_for(std::chrono::milliseconds(20)); }
        const auto view = link.view(r);
        for (int k = 0; k < kSteps; ++k) {
            const int s = (k * 7 + k / 3) % 3;
            Tensor p(staged[k][s], DType::BF16, {5120, columns[s]});
            Tensor x(residual[s], DType::BF16, {5120, columns[s]});
            ops::tp_residual_allreduce(p, x, view, stream);
        }
        check(cudaStreamSynchronize(stream), "chain");
        for (int s = 0; s < 3; ++s) {
            results[r][s].resize(std::size_t(5120) * columns[s]);
            check(cudaMemcpy(results[r][s].data(), residual[s], results[r][s].size() * 2,
                             cudaMemcpyDeviceToHost),
                  "download");
            check(cudaFree(residual[s]), "free");
            for (int k = 0; k < kSteps; ++k) check(cudaFree(staged[k][s]), "free");
        }
        check(cudaStreamDestroy(stream), "stream");
    });
    int failures = 0;
    for (int s = 0; s < 3; ++s) {
        std::vector<float> stored = residual0[s];
        for (int k = 0; k < kSteps; ++k) {
            if ((k * 7 + k / 3) % 3 != s) continue;
            for (std::size_t i = 0; i < stored.size(); ++i) {
                const float value = (stored[i] + partials[0][s][k][i]) + partials[1][s][k][i];
                stored[i]         = bf16_to_f32(f32_to_bf16(value));
            }
        }
        const auto expected = bf16_bits(stored);
        failures += verify_exact("mixed chain rank0", results[0][s], expected);
        failures += verify_exact("mixed chain rank1", results[1][s], expected);
    }
    return failures;
}

// Copy-channel collectives as the prefill pipeline issues them: a mixer stage of column-slice
// all-reduces left in flight, then an FFN stage that waits each mixer slice before issuing its
// own, with a synchronous decode-sized collective interleaved and rank 1 lagging. Results must
// match the synchronous collective's bits on both ranks.
int async_case(TensorParallelLink& link, const std::array<int, 2>& devices) {
    constexpr std::int32_t kRows    = 5120;
    constexpr std::int32_t kColumns = 2048;
    constexpr int kSlices           = 4;
    constexpr int kLayers           = 3;
    constexpr std::int32_t kSlice   = kColumns / kSlices;
    const std::size_t count         = std::size_t(kRows) * kColumns;
    std::vector<float> residual0(count);
    fill_uniform(residual0, 500, -1.0f, 1.0f);
    round_to_bf16(residual0);
    // partials[rank][stage]: stage 2l is layer l's mixer, 2l+1 its FFN.
    std::array<std::vector<std::vector<float>>, 2> partials;
    for (int r = 0; r < 2; ++r) {
        partials[r].resize(2 * kLayers);
        for (int k = 0; k < 2 * kLayers; ++k) {
            partials[r][k].resize(count);
            fill_uniform(partials[r][k], 600 + 13 * k + 1000 * r, -0.5f, 0.5f);
            round_to_bf16(partials[r][k]);
        }
    }
    std::array<std::vector<std::uint16_t>, 2> results;
    std::array<int, 2> asynchronous{};
    on_ranks(devices, [&](int r) {
        cudaStream_t stream = nullptr;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        void* residual = nullptr;
        void* small    = nullptr;
        check(cudaMalloc(&residual, count * 2), "residual");
        check(cudaMalloc(&small, kRows * 2), "small residual");
        check(cudaMemset(small, 0, kRows * 2), "zero small");
        const auto bits = bf16_bits(residual0);
        check(cudaMemcpy(residual, bits.data(), count * 2, cudaMemcpyHostToDevice), "upload");
        std::vector<void*> staged(2 * kLayers);
        for (int k = 0; k < 2 * kLayers; ++k) {
            check(cudaMalloc(&staged[k], count * 2), "partial");
            const auto pbits = bf16_bits(partials[r][k]);
            check(cudaMemcpy(staged[k], pbits.data(), count * 2, cudaMemcpyHostToDevice),
                  "upload partial");
        }
        check(cudaDeviceSynchronize(), "uploads");
        if (r == 1) { std::this_thread::sleep_for(std::chrono::milliseconds(20)); }
        const auto view = link.view(r);
        const auto slice = [&](void* base, int index) {
            return Tensor(static_cast<std::uint16_t*>(base) + std::size_t(kRows) * kSlice * index,
                          DType::BF16, {kRows, kSlice});
        };
        Tensor small_partial(staged[0], DType::BF16, {kRows, 1});
        Tensor small_residual(small, DType::BF16, {kRows, 1});
        for (int layer = 0; layer < kLayers; ++layer) {
            std::array<ops::TensorParallelTicket, kSlices> mixer{};
            for (int i = 0; i < kSlices; ++i) {
                Tensor x = slice(residual, i);
                mixer[i] = ops::tp_residual_allreduce_async(slice(staged[2 * layer], i), x, view,
                                                            stream);
                asynchronous[r] += mixer[i] != 0;
            }
            ops::tp_residual_allreduce(small_partial, small_residual, view, stream);
            std::array<ops::TensorParallelTicket, kSlices> ffn{};
            for (int i = 0; i < kSlices; ++i) {
                ops::tp_wait(view, mixer[i], stream);
                Tensor x = slice(residual, i);
                ffn[i]   = ops::tp_residual_allreduce_async(slice(staged[2 * layer + 1], i), x,
                                                            view, stream);
            }
            for (const auto ticket : ffn) ops::tp_wait(view, ticket, stream);
        }
        check(cudaStreamSynchronize(stream), "pipeline");
        results[r].resize(count);
        check(cudaMemcpy(results[r].data(), residual, count * 2, cudaMemcpyDeviceToHost),
              "download");
        for (void* p : staged) check(cudaFree(p), "free partial");
        check(cudaFree(residual), "free residual");
        check(cudaFree(small), "free small");
        check(cudaStreamDestroy(stream), "stream");
    });
    std::vector<float> stored = residual0;
    for (int k = 0; k < 2 * kLayers; ++k) {
        for (std::size_t i = 0; i < count; ++i) {
            const float value = (stored[i] + partials[0][k][i]) + partials[1][k][i];
            stored[i]         = bf16_to_f32(f32_to_bf16(value));
        }
    }
    const auto expected = bf16_bits(stored);
    int failures        = 0;
    if (asynchronous[0] != kSlices * kLayers || asynchronous[1] != kSlices * kLayers) {
        std::printf("FAIL async pipeline: copy channel not used (%d/%d)\n", asynchronous[0],
                    asynchronous[1]);
        ++failures;
    }
    failures += verify_exact("async pipeline rank0", results[0], expected);
    failures += verify_exact("async pipeline rank1", results[1], expected);
    return failures;
}

// Throughput of one 2048-column all-reduce, synchronous kernel versus copy channel.
int async_timing_case(TensorParallelLink& link, std::array<RankBuffers, 2>& ranks,
                      const std::array<int, 2>& devices) {
    constexpr int kCalls = 16;
    std::array<double, 2> sync_us{}, async_us{};
    on_ranks(devices, [&](int r) {
        auto& b         = ranks[r];
        const auto view = link.view(r);
        Tensor p(b.partial, DType::BF16, {5120, 2048});
        Tensor x(b.residual, DType::BF16, {5120, 2048});
        check(cudaMemsetAsync(b.partial, 0, p.bytes(), b.stream), "zero partial");
        check(cudaMemsetAsync(b.residual, 0, x.bytes(), b.stream), "zero residual");
        ops::tp_residual_allreduce(p, x, view, b.stream);
        ops::tp_wait(view, ops::tp_residual_allreduce_async(p, x, view, b.stream), b.stream);
        check(cudaStreamSynchronize(b.stream), "warm");
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < kCalls; ++i) ops::tp_residual_allreduce(p, x, view, b.stream);
        check(cudaStreamSynchronize(b.stream), "sync");
        sync_us[r] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() -
                                                               start)
                         .count() /
                     kCalls;
        start = std::chrono::steady_clock::now();
        for (int i = 0; i < kCalls; ++i) {
            ops::tp_wait(view, ops::tp_residual_allreduce_async(p, x, view, b.stream), b.stream);
        }
        check(cudaStreamSynchronize(b.stream), "async");
        async_us[r] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() -
                                                                start)
                          .count() /
                      kCalls;
    });
    std::printf("tp_residual_allreduce [5120,2048]: kernel %.1f us, copy channel %.1f us\n",
                sync_us[0], async_us[0]);
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
    failures += mixed_chain_case(link, devices);
    failures += residual_case(link, ranks, devices, 5120, 3, 9, 16);
    failures += async_case(link, devices);
    failures += async_timing_case(link, ranks, devices);
    failures += mixed_chain_case(link, devices);
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
