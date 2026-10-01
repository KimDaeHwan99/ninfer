#pragma once

// The host thread that drives the peer rank of a tensor-parallel Program. The two rank Programs
// receive the same calls; the peer's call runs on this thread, bound to the peer device, while
// the primary's runs on the caller. Collectives synchronize the two devices without host events,
// so both calls must be in flight together, and each may block on its own device independently.

#include "core/device.h"

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>

namespace ninfer::models::qwen3_5::detail {

class PeerExecutor {
public:
    explicit PeerExecutor(const DeviceContext& device);
    ~PeerExecutor();

    PeerExecutor(const PeerExecutor&)            = delete;
    PeerExecutor& operator=(const PeerExecutor&) = delete;

    // Runs `peer` on the peer thread and `local` on the caller, returning after both finished.
    // The caller's exception is rethrown first, then the peer's.
    void run(const std::function<void()>& peer, const std::function<void()>& local);

private:
    void loop(const DeviceContext& device);

    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_;
    const std::function<void()>* task_ = nullptr;
    std::exception_ptr error_;
    bool finished_ = false;
    bool stop_     = false;
    std::thread thread_;
};

} // namespace ninfer::models::qwen3_5::detail
