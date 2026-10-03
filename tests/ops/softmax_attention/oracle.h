#pragma once

#include "ninfer/ops/attention_geometry.h"
#include "ops/host_parallel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ninfer::test {

// Independent logical Softmax Attention oracle. Callbacks expose represented public values and
// the entry-specific visible set; no production staging cast, tile, cache address, or reduction
// tree is reproduced here. Each (query, head) row is evaluated in the same order whatever the
// thread count, so large cases split rows across host threads without changing a single bit;
// callbacks must therefore be pure reads, and `store` writes one distinct element per call.
template <typename QueryValue, typename KeyValue, typename ValueValue, typename Visible,
          typename Store>
void naive_dense_softmax_attention(ops::AttentionHeadGeometry geometry, int query_tokens,
                                   int key_tokens, double scale, QueryValue query_value,
                                   KeyValue key_value, ValueValue value_value, Visible visible,
                                   Store store) {
    if (!ops::valid_attention_head_geometry(geometry) || query_tokens < 0 || key_tokens < 0) {
        throw std::invalid_argument("invalid naive Softmax Attention geometry");
    }
    const int group          = geometry.query_heads / geometry.kv_heads;
    const std::int64_t rows  = static_cast<std::int64_t>(query_tokens) * geometry.query_heads;
    const auto evaluate_rows = [&](std::int64_t begin, std::int64_t end) {
        const auto head_dim = static_cast<std::size_t>(geometry.head_dim);
        std::vector<double> scores(static_cast<std::size_t>(key_tokens));
        std::vector<double> query_row(head_dim), numerators(head_dim);
        for (std::int64_t row = begin; row < end; ++row) {
            const int query      = static_cast<int>(row / geometry.query_heads);
            const int query_head = static_cast<int>(row % geometry.query_heads);
            const int kv_head    = query_head / group;
            for (int d = 0; d < geometry.head_dim; ++d) {
                query_row[static_cast<std::size_t>(d)] = query_value(d, query_head, query);
            }
            double maximum = -std::numeric_limits<double>::infinity();
            for (int key = 0; key < key_tokens; ++key) {
                if (!visible(query, key)) {
                    scores[static_cast<std::size_t>(key)] =
                        -std::numeric_limits<double>::infinity();
                    continue;
                }
                double dot = 0.0;
                for (int d = 0; d < geometry.head_dim; ++d) {
                    dot += query_row[static_cast<std::size_t>(d)] * key_value(d, kv_head, key);
                }
                const double score                    = dot * scale;
                scores[static_cast<std::size_t>(key)] = score;
                maximum                               = std::max(maximum, score);
            }

            double denominator = 0.0;
            if (maximum != -std::numeric_limits<double>::infinity()) {
                for (int key = 0; key < key_tokens; ++key) {
                    double& score = scores[static_cast<std::size_t>(key)];
                    if (score == -std::numeric_limits<double>::infinity()) continue;
                    score = std::exp(score - maximum);
                    denominator += score;
                }
            }
            // Key-outer accumulation keeps every channel's own key order and walks V contiguously.
            std::fill(numerators.begin(), numerators.end(), 0.0);
            for (int key = 0; key < key_tokens; ++key) {
                const double weight = scores[static_cast<std::size_t>(key)];
                if (weight == -std::numeric_limits<double>::infinity()) continue;
                for (int d = 0; d < geometry.head_dim; ++d) {
                    numerators[static_cast<std::size_t>(d)] +=
                        weight * value_value(d, kv_head, key);
                }
            }
            for (int d = 0; d < geometry.head_dim; ++d) {
                const double numerator = numerators[static_cast<std::size_t>(d)];
                store(d, query_head, query, denominator > 0.0 ? numerator / denominator : 0.0);
            }
        }
    };

    // Small cases are not worth a thread; large ones (production widths over long histories)
    // take minutes on one core.
    const std::uint64_t work = static_cast<std::uint64_t>(rows) *
                               static_cast<std::uint64_t>(key_tokens) *
                               static_cast<std::uint64_t>(geometry.head_dim);
    parallel_ranges(rows, work < (1ULL << 24) ? 1 : host_thread_count(), evaluate_rows);
}

} // namespace ninfer::test
