// Tensor-parallel shard planning. The full model is bound first, so the artifact is validated
// against the complete mathematics; this file then rewrites the bound plan into one rank's share:
// a rank-local config, shard-shaped physical placements whose bytes are strided windows of the
// stored objects, and logical parameters re-pointed into those shards.
//
// Split map (Megatron-style, per rank r of S):
//   column-parallel (rows kept):  attention query/gate (query heads), key/value (KV heads),
//                                 GDN query/key (key heads), value/z, a/b projections, a_log,
//                                 dt_bias (value heads), MLP gate/up, text output and proposal
//                                 heads (vocabulary rows; logits are gathered).
//   row-parallel (columns kept):  attention/GDN output, MLP down; the rank's partial products are
//                                 summed by the residual all-reduce.
//   channel-split (columns kept): GDN convolution, whose channels are Q | K | V sections.
//   replicated:                   norms, token embedding, MTP input projection, proposal ids.
// Head-aligned row ranges keep every split exact; nothing is repacked.
#include "models/qwen3_5/load/bindings.h"

#include "artifact/reader.h"

#include <algorithm>
#include <map>
#include <string>
#include <string_view>

namespace ninfer::models::qwen3_5::loading {
namespace {

using artifact::ArtifactError;
using artifact::ByteWindow;
using artifact::Shape;

enum class Axis : std::uint8_t { Replicated, Rows, Columns };

struct Interval {
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;

    [[nodiscard]] std::uint64_t size() const noexcept { return end - begin; }
};

struct ParameterSplit {
    Axis axis = Axis::Replicated;
    std::vector<Interval> keep; // Parameter rows (Rows) or columns (Columns).
};

bool ends_with(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
}

std::uint64_t divide(std::uint64_t value, std::uint32_t parts, const std::string& what) {
    if (value == 0 || value % parts) {
        throw ArtifactError("tensor parallel: " + what + " does not split evenly");
    }
    return value / parts;
}

Interval share(std::uint64_t extent, TensorParallelPlacement tp, const std::string& what) {
    const auto part = divide(extent, tp.size, what);
    return {part * tp.rank, part * (tp.rank + 1)};
}

std::uint64_t rows_of(const Shape& shape) { return shape.empty() ? 1 : shape.front(); }

std::uint64_t columns_of(const Shape& shape) { return shape.size() == 2 ? shape[1] : 1; }

ParameterSplit split_for(const std::string& name, const Shape& shape, const Config& full,
                         TensorParallelPlacement tp) {
    const auto rows = [&](std::uint64_t heads, std::uint64_t head_width) {
        // Head-aligned share of a [heads * head_width, ...] parameter.
        const auto heads_per_rank = divide(heads, tp.size, name + " heads");
        if (rows_of(shape) != heads * head_width) {
            throw ArtifactError("tensor parallel: " + name + " rows differ from its heads");
        }
        return ParameterSplit{Axis::Rows,
                              {{heads_per_rank * head_width * tp.rank,
                                heads_per_rank * head_width * (tp.rank + 1)}}};
    };
    const auto columns = [&] {
        return ParameterSplit{Axis::Columns, {share(columns_of(shape), tp, name + " columns")}};
    };
    const auto replicated = ParameterSplit{};
    const auto& text      = full.text;
    if (ends_with(name, "/attention/query") || ends_with(name, "/attention/gate")) {
        return rows(text.attention->num_attention_heads, text.attention->head_dim);
    }
    if (ends_with(name, "/attention/key") || ends_with(name, "/attention/value")) {
        return rows(text.attention->num_key_value_heads, text.attention->head_dim);
    }
    if (ends_with(name, "/attention/output") || ends_with(name, "/gdn/output") ||
        ends_with(name, "/mlp/down")) {
        return columns();
    }
    if (ends_with(name, "/attention/query_norm") || ends_with(name, "/attention/key_norm") ||
        ends_with(name, "/gdn/norm") || ends_with(name, "input_norm") ||
        ends_with(name, "post_attention_norm") || ends_with(name, "final_norm") ||
        ends_with(name, "embedding_norm") || ends_with(name, "hidden_norm")) {
        return replicated;
    }
    if (ends_with(name, "/gdn/query") || ends_with(name, "/gdn/key")) {
        return rows(text.gdn->linear_num_key_heads, text.gdn->linear_key_head_dim);
    }
    if (ends_with(name, "/gdn/value") || ends_with(name, "/gdn/z")) {
        return rows(text.gdn->linear_num_value_heads, text.gdn->linear_value_head_dim);
    }
    if (ends_with(name, "/gdn/a_projection") || ends_with(name, "/gdn/b_projection") ||
        ends_with(name, "/gdn/a_log") || ends_with(name, "/gdn/dt_bias")) {
        return rows(text.gdn->linear_num_value_heads, 1);
    }
    if (ends_with(name, "/gdn/convolution")) {
        const auto key   = text.gdn->key_width();
        const auto value = text.gdn->value_width();
        if (shape.size() != 2 || shape[1] != 2 * key + value) {
            throw ArtifactError("tensor parallel: GDN convolution channels differ from Q|K|V");
        }
        const auto q = share(key, tp, name + " query channels");
        const auto v = share(value, tp, name + " value channels");
        return {Axis::Columns,
                {q, {key + q.begin, key + q.end}, {2 * key + v.begin, 2 * key + v.end}}};
    }
    if (ends_with(name, "/mlp/gate") || ends_with(name, "/mlp/up")) {
        return {Axis::Rows, {share(rows_of(shape), tp, name + " rows")}};
    }
    if (name == "text/output_head" || name == "proposal/head") {
        return {Axis::Rows, {share(rows_of(shape), tp, name + " vocabulary")}};
    }
    if (name == "text/token_embedding" || name == "mtp/input_projection" ||
        name == "proposal/token_ids") {
        return replicated;
    }
    throw ArtifactError("tensor parallel: no split rule for parameter " + name);
}

// One physical object's rank share: kept object rows or columns, in object order.
struct ObjectShard {
    Axis axis = Axis::Replicated;
    std::vector<Interval> keep;
    bool replicated_use = false;
};

void add_interval(std::vector<Interval>& intervals, Interval value) {
    if (value.size() == 0) { return; }
    intervals.push_back(value);
}

void normalize(std::vector<Interval>& intervals, const std::string& object) {
    std::sort(intervals.begin(), intervals.end(),
              [](const Interval& a, const Interval& b) { return a.begin < b.begin; });
    std::vector<Interval> merged;
    for (const auto& value : intervals) {
        if (!merged.empty() && value.begin < merged.back().end) {
            throw ArtifactError("tensor parallel: overlapping shares of " + object);
        }
        if (!merged.empty() && value.begin == merged.back().end) {
            merged.back().end = value.end;
        } else {
            merged.push_back(value);
        }
    }
    intervals = std::move(merged);
}

std::uint64_t total(const std::vector<Interval>& intervals) {
    std::uint64_t out = 0;
    for (const auto& value : intervals) { out += value.size(); }
    return out;
}

// Shard row/column index of an object row/column inside the kept intervals.
std::uint64_t shard_index(const std::vector<Interval>& keep, std::uint64_t index) {
    std::uint64_t base = 0;
    for (const auto& value : keep) {
        if (index >= value.begin && index < value.end) { return base + (index - value.begin); }
        base += value.size();
    }
    throw std::logic_error("tensor parallel: index outside its object share");
}

void require_aligned(const std::vector<Interval>& keep, std::uint64_t alignment,
                     const std::string& object, const char* what) {
    for (const auto& value : keep) {
        if (value.begin % alignment || value.end % alignment) {
            throw ArtifactError("tensor parallel: " + object + " " + what +
                                " share is not aligned to " + std::to_string(alignment));
        }
    }
}

struct PlacedShard {
    WeightGeometry geometry;
    std::vector<ByteWindow> windows;
};

// Byte windows that copy one plane's kept rows (row bytes `row_bytes`) or kept columns.
void row_windows(std::vector<ByteWindow>& out, const std::vector<Interval>& keep,
                 std::uint64_t source_plane, std::uint64_t destination_plane,
                 std::uint64_t row_bytes) {
    std::uint64_t cursor = 0;
    for (const auto& value : keep) {
        out.push_back({.source_offset      = source_plane + value.begin * row_bytes,
                       .destination_offset = destination_plane + cursor * row_bytes,
                       .width              = value.size() * row_bytes});
        cursor += value.size();
    }
}

void column_windows(std::vector<ByteWindow>& out, const std::vector<Interval>& keep,
                    std::uint64_t source_plane, std::uint64_t source_pitch,
                    std::uint64_t destination_plane, std::uint64_t destination_pitch,
                    std::uint64_t unit_columns, std::uint64_t unit_bytes, std::uint64_t height) {
    std::uint64_t cursor = 0;
    for (const auto& value : keep) {
        out.push_back({.source_offset      = source_plane + value.begin / unit_columns * unit_bytes,
                       .source_pitch       = source_pitch,
                       .destination_offset = destination_plane + cursor / unit_columns * unit_bytes,
                       .destination_pitch  = destination_pitch,
                       .width              = value.size() / unit_columns * unit_bytes,
                       .height             = height});
        cursor += value.size();
    }
}

PlacedShard place_shard(const WeightGeometry& source, const ObjectShard& shard,
                        const std::string& object) {
    Shape shape = source.shape;
    if (shape.empty() || shape.size() > 2) {
        throw ArtifactError("tensor parallel: " + object + " is not a vector or matrix");
    }
    const auto rows    = shape[0];
    const auto columns = shape.size() == 2 ? shape[1] : 1;
    if (shard.axis == Axis::Rows) {
        shape[0] = total(shard.keep);
    } else {
        if (shape.size() != 2) {
            throw ArtifactError("tensor parallel: column share of vector " + object);
        }
        shape[1] = total(shard.keep);
    }
    PlacedShard out;
    out.geometry = weight_geometry(source.format, source.layout, shape);
    const auto& g = out.geometry;
    switch (source.layout) {
    case QuantLayout::Contiguous: {
        const auto word = source.code_bytes / source.elements;
        if (shard.axis == Axis::Rows) {
            row_windows(out.windows, shard.keep, 0, 0, columns * word);
        } else {
            column_windows(out.windows, shard.keep, 0, columns * word, 0, shape[1] * word, 1, word,
                           rows);
        }
        break;
    }
    case QuantLayout::RowScale: {
        if (shard.axis == Axis::Rows) {
            row_windows(out.windows, shard.keep, 0, 0, columns);
            row_windows(out.windows, shard.keep, source.scale_offset, g.scale_offset, 2);
        } else {
            column_windows(out.windows, shard.keep, 0, columns, 0, shape[1], 1, 1, rows);
            out.windows.push_back({.source_offset      = source.scale_offset,
                                   .destination_offset = g.scale_offset,
                                   .width              = source.scale_bytes});
        }
        break;
    }
    case QuantLayout::RowSplit: {
        if (shard.axis == Axis::Rows) {
            row_windows(out.windows, shard.keep, 0, 0, source.code_bytes_per_row);
            if (source.high_bytes_per_row) {
                row_windows(out.windows, shard.keep, source.high_offset, g.high_offset,
                            source.high_bytes_per_row);
            }
            row_windows(out.windows, shard.keep, source.scale_offset, g.scale_offset,
                        source.scale_bytes_per_row);
        } else {
            // A 128-column share keeps whole groups and leaves the shard without padding.
            if (source.padded_columns != columns) {
                throw ArtifactError("tensor parallel: padded RowSplit " + object +
                                    " has no column share");
            }
            require_aligned(shard.keep, 128, object, "column");
            const auto groups = source.padded_columns / source.group_size;
            column_windows(out.windows, shard.keep, 0, source.code_bytes_per_row, 0,
                           g.code_bytes_per_row, source.group_size,
                           source.code_bytes_per_row / groups, rows);
            if (source.high_bytes_per_row) {
                column_windows(out.windows, shard.keep, source.high_offset,
                               source.high_bytes_per_row, g.high_offset, g.high_bytes_per_row,
                               source.group_size, source.high_bytes_per_row / groups, rows);
            }
            column_windows(out.windows, shard.keep, source.scale_offset,
                           source.scale_bytes_per_row, g.scale_offset, g.scale_bytes_per_row,
                           source.group_size, source.scale_bytes_per_row / groups, rows);
        }
        break;
    }
    case QuantLayout::BlockScaleK16M128x4: {
        // Scales are 512-byte tiles of 128 rows x 64 columns, tile-row major.
        const auto tiles        = columns / 64;
        const auto shard_tiles  = shape[1] / 64;
        constexpr auto kTile    = std::uint64_t{512};
        if (shard.axis == Axis::Rows) {
            require_aligned(shard.keep, 128, object, "row");
            row_windows(out.windows, shard.keep, 0, 0, columns / 2);
            std::vector<Interval> tile_rows;
            for (const auto& value : shard.keep) {
                tile_rows.push_back({value.begin / 128, value.end / 128});
            }
            row_windows(out.windows, tile_rows, source.scale_offset, g.scale_offset,
                        tiles * kTile);
        } else {
            require_aligned(shard.keep, 64, object, "column");
            column_windows(out.windows, shard.keep, 0, columns / 2, 0, shape[1] / 2, 2, 1, rows);
            column_windows(out.windows, shard.keep, source.scale_offset, tiles * kTile,
                           g.scale_offset, shard_tiles * kTile, 64, kTile, rows / 128);
        }
        out.windows.push_back({.source_offset      = source.divisor_offset,
                               .destination_offset = g.divisor_offset,
                               .width              = 4});
        break;
    }
    }
    return out;
}

} // namespace

Config shard_config(const Config& full, TensorParallelPlacement tp) {
    if (!tp.split()) { return full; }
    if (tp.size != 2 || tp.rank >= tp.size) {
        throw ArtifactError("tensor parallel: only a two-way split is implemented");
    }
    if (full.text.architecture != Architecture::Qwen3_5 ||
        !std::holds_alternative<DenseConfig>(full.text.ffn) || !full.text.attention ||
        !full.text.gdn) {
        throw ArtifactError("tensor parallel: requires the dense Qwen3.5 text architecture");
    }
    if (full.vision) { throw ArtifactError("tensor parallel: Vision is not split"); }
    if (full.draft) { throw ArtifactError("tensor parallel: DFlash drafts are not split"); }
    Config out          = full;
    auto& attention     = *out.text.attention;
    auto& gdn           = *out.text.gdn;
    auto& dense         = std::get<DenseConfig>(out.text.ffn);
    const auto local    = [&](std::uint32_t value, const char* what) {
        return static_cast<std::uint32_t>(divide(value, tp.size, what));
    };
    attention.num_attention_heads = local(attention.num_attention_heads, "attention heads");
    attention.num_key_value_heads = local(attention.num_key_value_heads, "KV heads");
    gdn.linear_num_key_heads      = local(gdn.linear_num_key_heads, "GDN key heads");
    gdn.linear_num_value_heads    = local(gdn.linear_num_value_heads, "GDN value heads");
    dense.intermediate_size       = local(dense.intermediate_size, "FFN width");
    if (out.text.vocab_size % (8 * tp.size)) {
        throw ArtifactError("tensor parallel: vocabulary does not split into BF16x8 rows");
    }
    out.tensor_parallel = tp;
    return out;
}

void shard_plan(const artifact::Reader& reader, const Config& full,
                std::vector<PendingWeight>& pending,
                artifact::MaterializationPlan& materialization, TensorParallelPlacement tp) {
    if (!tp.split()) { return; }
    std::vector<ParameterSplit> splits;
    splits.reserve(pending.size());
    std::map<std::size_t, ObjectShard> objects;
    const auto object_name = [&](artifact::ObjectHandle handle) {
        return artifact::object_id(reader.directory().object(handle));
    };
    for (const auto& weight : pending) {
        const auto& reference = weight.reference;
        auto split            = split_for(reference.name, reference.shape, full, tp);
        const auto columns    = columns_of(reference.shape);
        std::uint64_t parameter_row = 0;
        for (const auto& part : reference.binding.parts) {
            auto& shard          = objects[part.object.index];
            const auto& geometry = reader.geometry(part.object);
            if (split.axis == Axis::Replicated) {
                shard.replicated_use = true;
                continue;
            }
            if (shard.axis != Axis::Replicated && shard.axis != split.axis) {
                throw ArtifactError("tensor parallel: " + object_name(part.object) +
                                    " is split along two axes");
            }
            shard.axis = split.axis;
            if (split.axis == Axis::Columns) {
                if (reference.binding.parts.size() != 1 || part.begin != 0 ||
                    part.end != geometry.elements || geometry.shape != reference.shape) {
                    throw ArtifactError("tensor parallel: column share of " + reference.name +
                                        " requires its whole object");
                }
                if (!shard.keep.empty()) {
                    throw ArtifactError("tensor parallel: shared column-split object " +
                                        object_name(part.object));
                }
                shard.keep = split.keep;
                continue;
            }
            const auto object_columns = columns_of(geometry.shape);
            if (object_columns != columns || part.begin % columns || part.end % columns) {
                throw ArtifactError("tensor parallel: " + reference.name +
                                    " does not cover whole object rows");
            }
            const Interval covered{parameter_row, parameter_row + (part.end - part.begin) / columns};
            const auto object_row = part.begin / columns;
            for (const auto& keep : split.keep) {
                const auto begin = std::max(keep.begin, covered.begin);
                const auto end   = std::min(keep.end, covered.end);
                if (begin < end) {
                    add_interval(shard.keep, {object_row + (begin - covered.begin),
                                              object_row + (end - covered.begin)});
                }
            }
            parameter_row = covered.end;
        }
        splits.push_back(std::move(split));
    }
    for (auto& [index, shard] : objects) {
        if (shard.axis == Axis::Replicated) { continue; }
        const auto name = object_name(artifact::ObjectHandle{index});
        if (shard.replicated_use) {
            throw ArtifactError("tensor parallel: " + name + " is both split and replicated");
        }
        normalize(shard.keep, name);
    }

    // Re-point every split parameter into its object's shard.
    for (std::size_t i = 0; i < pending.size(); ++i) {
        auto& reference    = pending[i].reference;
        const auto& split  = splits[i];
        if (split.axis == Axis::Replicated) { continue; }
        const auto columns = columns_of(reference.shape);
        std::vector<artifact::Part> parts;
        Shape shape = reference.shape;
        if (split.axis == Axis::Columns) {
            const auto& part = reference.binding.parts.front();
            shape[1]         = total(split.keep);
            parts.push_back({part.object, 0, artifact::checked_mul(shape[0], shape[1], "shard")});
        } else {
            const auto object_columns = columns;
            std::uint64_t parameter_row = 0;
            for (const auto& part : reference.binding.parts) {
                const auto& shard = objects.at(part.object.index);
                const Interval covered{parameter_row,
                                       parameter_row + (part.end - part.begin) / columns};
                const auto object_row = part.begin / columns;
                for (const auto& keep : split.keep) {
                    const auto begin = std::max(keep.begin, covered.begin);
                    const auto end   = std::min(keep.end, covered.end);
                    if (begin >= end) { continue; }
                    const auto first = shard_index(shard.keep, object_row + (begin - covered.begin));
                    const artifact::Part next{part.object, first * object_columns,
                                              (first + (end - begin)) * object_columns};
                    if (!parts.empty() && parts.back().object == next.object &&
                        parts.back().end == next.begin) {
                        parts.back().end = next.end;
                    } else {
                        parts.push_back(next);
                    }
                }
                parameter_row = covered.end;
            }
            shape[0] = total(split.keep);
        }
        reference.shape            = shape;
        reference.binding.parts    = std::move(parts);
        reference.binding.elements = artifact::checked_mul(rows_of(shape), columns_of(shape),
                                                           reference.name);
        reference.binding.whole_object = false;
    }

    // Shard-shaped placements, re-laid out in the rank's arena.
    std::uint64_t capacity = 0;
    for (auto& placement : materialization.device_objects) {
        const auto found = objects.find(placement.object.index);
        if (found != objects.end() && found->second.axis != Axis::Replicated) {
            auto placed = place_shard(reader.geometry(placement.object), found->second,
                                      object_name(placement.object));
            placement.bytes  = placed.geometry.bytes;
            placement.subset = artifact::PlacementSubset{std::move(placed.geometry),
                                                         std::move(placed.windows)};
        }
        placement.offset = artifact::align_up(capacity, placement.alignment, "shard offset");
        capacity         = artifact::checked_add(placement.offset, placement.bytes, "shard arena");
    }
    materialization.device_capacity_bytes = capacity;
    for (const auto& [index, shard] : objects) {
        if (shard.axis == Axis::Replicated) { continue; }
        const bool placed = std::any_of(
            materialization.device_objects.begin(), materialization.device_objects.end(),
            [&](const auto& placement) { return placement.object.index == index; });
        if (!placed) {
            throw ArtifactError("tensor parallel: split object " +
                                object_name(artifact::ObjectHandle{index}) +
                                " has no device placement");
        }
    }
}

} // namespace ninfer::models::qwen3_5::loading
