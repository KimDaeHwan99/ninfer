#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::product {

// `--devices A,B`: one non-negative CUDA device id per tensor-parallel rank.
[[nodiscard]] inline std::vector<int> parse_device_list(std::string_view text) {
    std::vector<int> out;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const auto end   = text.find(',', begin);
        const auto token = text.substr(begin, end == std::string_view::npos ? text.size() - begin
                                                                            : end - begin);
        if (token.empty() || token.size() > 4 ||
            token.find_first_not_of("0123456789") != std::string_view::npos) {
            throw std::invalid_argument("--devices expects comma-separated device ids: " +
                                        std::string(text));
        }
        out.push_back(std::stoi(std::string(token)));
        if (end == std::string_view::npos) { break; }
        begin = end + 1;
    }
    return out;
}

[[nodiscard]] inline std::uint32_t parse_tensor_parallel(std::string_view text) {
    if (text == "1") { return 1; }
    if (text == "2") { return 2; }
    throw std::invalid_argument("--tp must be 1 or 2");
}

// Resolves `--tp`, `--devices` and `--device` into the per-rank device list. `--tp 2` requires
// an explicit `--devices` naming two distinct devices; `--device`, when also given, must name the
// primary (rank 0) device.
[[nodiscard]] inline std::vector<int> resolve_rank_devices(std::uint32_t tensor_parallel,
                                                           const std::vector<int>& devices,
                                                           std::optional<int> device) {
    if (devices.empty()) {
        if (tensor_parallel != 1) {
            throw std::invalid_argument("--tp 2 requires --devices A,B");
        }
        return {device.value_or(0)};
    }
    if (devices.size() != tensor_parallel) {
        throw std::invalid_argument("--devices must list exactly one device per --tp rank");
    }
    if (tensor_parallel == 2 && devices[0] == devices[1]) {
        throw std::invalid_argument("--devices must name two distinct devices");
    }
    if (device && *device != devices[0]) {
        throw std::invalid_argument("--device and --devices disagree on the primary device");
    }
    return devices;
}

} // namespace ninfer::product
