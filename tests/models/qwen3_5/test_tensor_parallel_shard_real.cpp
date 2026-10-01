// Host-only qualification of the tensor-parallel shard plan against a real artifact: for every
// logical parameter of both ranks, sampled elements decoded from the rank's shard bytes (produced
// only through the placement windows) must equal the corresponding elements of the full model,
// with the expected global index derived independently from the Megatron split.
//
//   ninfer_qwen3_5_tensor_parallel_shard_real_test --artifact model.ninfer
#include "artifact/binder.h"
#include "artifact/reader.h"
#include "models/qwen3_5/load.h"
#include "ops/quantized_weight.h"

#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using namespace ninfer;
namespace qwen = ninfer::models::qwen3_5;
namespace qd   = ninfer::test::quantized_weight::detail;

struct Source {
    const artifact::Reader* reader          = nullptr;
    artifact::ObjectHandle object;
    WeightGeometry geometry;
    const artifact::PlacementSubset* subset = nullptr;

    // Byte of the placed representation, read through the placement windows.
    std::uint8_t byte(std::uint64_t offset) const {
        std::uint64_t source = offset;
        if (subset != nullptr) {
            bool found = false;
            for (const auto& w : subset->windows) {
                if (offset < w.destination_offset) { continue; }
                const auto delta = offset - w.destination_offset;
                const auto row   = w.destination_pitch ? delta / w.destination_pitch : 0;
                const auto col   = delta - row * w.destination_pitch;
                if (row < w.height && col < w.width) {
                    source = w.source_offset + row * w.source_pitch + col;
                    found  = true;
                    break;
                }
            }
            if (!found) { throw std::runtime_error("shard byte not covered by any window"); }
        }
        const auto& descriptor = reader->directory().tensor(object);
        std::byte value{};
        reader->read_into(descriptor.offset + source, {&value, 1});
        return static_cast<std::uint8_t>(value);
    }

    std::uint16_t u16(std::uint64_t offset) const {
        return static_cast<std::uint16_t>(byte(offset) | (byte(offset + 1) << 8));
    }

    std::uint32_t u32(std::uint64_t offset) const {
        return u16(offset) | (static_cast<std::uint32_t>(u16(offset + 2)) << 16);
    }

    double decode(std::uint64_t row, std::uint64_t column) const {
        const auto& g      = geometry;
        const auto columns = g.shape.size() == 2 ? g.shape[1] : 1;
        switch (g.layout) {
        case QuantLayout::Contiguous: {
            const auto index = row * columns + column;
            if (g.format == QType::BF16) { return qd::bf16_to_f32(u16(index * 2)); }
            const auto bits = u32(index * 4);
            float value;
            std::memcpy(&value, &bits, 4);
            return value;
        }
        case QuantLayout::RowScale:
            return qd::decode_e4m3fn(byte(row * columns + column)) *
                   qd::bf16_to_f32(u16(g.scale_offset + row * 2));
        case QuantLayout::BlockScaleK16M128x4: {
            const auto packed = byte(row * columns / 2 + column / 2);
            const auto code   = (column & 1) ? (packed >> 4) : (packed & 0x0f);
            const auto bits   = u32(g.divisor_offset);
            float divisor;
            std::memcpy(&divisor, &bits, 4);
            return qd::decode_e2m1(static_cast<std::uint8_t>(code)) *
                   qd::decode_e4m3fn(byte(weight_scale_offset(g, row, column / 16))) /
                   divisor;
        }
        case QuantLayout::RowSplit: {
            const auto spec   = qd::quant_spec(g.format);
            const auto nib    = qd::nibble_bytes_per_group(spec);
            const auto hb     = qd::high_bytes_per_group(spec);
            const auto groups = g.padded_columns / g.group_size;
            const auto group  = row * groups + column / g.group_size;
            std::uint8_t nibble[64]{};
            std::uint8_t high[64]{};
            for (int i = 0; i < nib; ++i) nibble[i] = byte(group * nib + i);
            for (int i = 0; i < hb; ++i) high[i] = byte(g.high_offset + group * hb + i);
            const int code = qd::unpack_lowbit_code(
                nibble, hb ? high : nullptr, spec, static_cast<int>(column % g.group_size));
            return code * static_cast<double>(qd::f16_to_f32(u16(g.scale_offset +
                                                                            group * 2)));
        }
        }
        throw std::logic_error("unknown layout");
    }
};

struct Plan {
    qwen::LoadPlan load;

    Source source(const artifact::Reader& reader, artifact::ObjectHandle object) const {
        Source out{&reader, object, reader.geometry(object), nullptr};
        for (const auto& p : load.materialization().device_objects) {
            if (p.object == object && p.subset) {
                out.geometry = p.subset->geometry;
                out.subset   = &*p.subset;
            }
        }
        return out;
    }

    // Decoded logical element of a parameter.
    double element(const artifact::Reader& reader, const artifact::ParameterReference& reference,
                   std::uint64_t row, std::uint64_t column) const {
        const auto columns = reference.shape.size() == 2 ? reference.shape[1] : 1;
        auto index         = row * columns + column;
        for (const auto& part : reference.binding.parts) {
            const auto size = part.end - part.begin;
            if (index < size) {
                const auto s        = source(reader, part.object);
                const auto parent   = part.begin + index;
                const auto pcolumns = s.geometry.shape.size() == 2 ? s.geometry.shape[1] : 1;
                return s.decode(parent / pcolumns, parent % pcolumns);
            }
            index -= size;
        }
        throw std::out_of_range("element outside parameter coverage");
    }
};

bool ends_with(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
}

int check(const artifact::Reader& reader, models::LoadOptions options) {
    const Plan full{qwen::plan_load(reader, options)};
    int failures        = 0;
    std::size_t checked = 0;
    for (std::uint32_t rank = 0; rank < 2; ++rank) {
        const Plan shard{qwen::plan_load(reader, options, {2, rank})};
        if (shard.load.parameter_count() != full.load.parameter_count()) {
            std::cerr << "parameter count differs\n";
            return 1;
        }
        const auto& text = shard.load.config().text;
        if (shard.load.config().tensor_parallel.rank != rank ||
            text.attention->num_attention_heads * 2 !=
                full.load.config().text.attention->num_attention_heads) {
            std::cerr << "rank config is not the local half\n";
            return 1;
        }
        std::mt19937_64 random(17 + rank);
        for (std::size_t i = 0; i < shard.load.parameter_count(); ++i) {
            const auto& local  = shard.load.parameter(qwen::WeightId{i});
            const auto& global = full.load.parameter(qwen::WeightId{i});
            if (local.residency != artifact::Residency::Device) { continue; }
            const auto lrows = local.shape.front();
            const auto lcols = local.shape.size() == 2 ? local.shape[1] : 1;
            const auto gcols = global.shape.size() == 2 ? global.shape[1] : 1;
            const bool rows_split    = lrows != global.shape.front();
            const bool columns_split = lcols != gcols;
            if (rows_split && columns_split) {
                std::cerr << local.name << ": split along both axes\n";
                ++failures;
                continue;
            }
            const auto map_column = [&](std::uint64_t column) {
                if (!columns_split) { return column; }
                if (ends_with(local.name, "/gdn/convolution")) {
                    const auto key   = full.load.config().text.gdn->key_width();
                    const auto local_key = key / 2;
                    if (column < local_key) { return rank * local_key + column; }
                    if (column < 2 * local_key) {
                        return key + rank * local_key + (column - local_key);
                    }
                    const auto value = full.load.config().text.gdn->value_width() / 2;
                    return 2 * key + rank * value + (column - 2 * local_key);
                }
                return rank * lcols + column;
            };
            const auto map_row = [&](std::uint64_t row) {
                return rows_split ? rank * lrows + row : row;
            };
            const auto probe = [&](std::uint64_t row, std::uint64_t column) {
                const double got  = shard.element(reader, local, row, column);
                const double want = full.element(reader, global, map_row(row), map_column(column));
                ++checked;
                if (!(got == want) && !(std::isnan(got) && std::isnan(want))) {
                    if (failures < 20) {
                        std::cerr << local.name << " rank " << rank << " [" << row << ","
                                  << column << "]: " << got << " != " << want << "\n";
                    }
                    ++failures;
                }
            };
            probe(0, 0);
            probe(lrows - 1, lcols - 1);
            probe(lrows - 1, 0);
            probe(0, lcols - 1);
            const int samples = (rows_split || columns_split) ? 48 : 8;
            for (int s = 0; s < samples; ++s) { probe(random() % lrows, random() % lcols); }
        }
    }
    std::cout << "checked " << checked << " sampled elements, " << failures << " mismatches\n";
    return failures ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    std::string path;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string_view(argv[i]) == "--artifact") { path = argv[i + 1]; }
    }
    if (path.empty()) {
        std::cout << "SKIP: --artifact model.ninfer not given\n";
        return 77;
    }
    try {
        artifact::Reader reader(path);
        int failures = 0;
        failures += check(reader, {});
        failures += check(reader, {.speculative   = SpeculativeBackend::Mtp,
                                   .proposal_head = ProposalHead::Optimized});
        std::cout << (failures ? "FAILED\n" : "PASSED\n");
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
