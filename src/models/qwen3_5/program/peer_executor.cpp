#include "models/qwen3_5/program/peer_executor.h"

namespace ninfer::models::qwen3_5::detail {

PeerExecutor::PeerExecutor(const DeviceContext& device)
    : thread_([this, &device] { loop(device); }) {}

PeerExecutor::~PeerExecutor() {
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
    }
    wake_.notify_one();
    thread_.join();
}

void PeerExecutor::loop(const DeviceContext& device) {
    device.bind_to_current_thread_noexcept();
    std::unique_lock lock(mutex_);
    for (;;) {
        wake_.wait(lock, [this] { return stop_ || task_ != nullptr; });
        if (task_ == nullptr) { return; }
        const auto* task = task_;
        lock.unlock();
        std::exception_ptr error;
        try {
            (*task)();
        } catch (...) {
            error = std::current_exception();
        }
        lock.lock();
        task_     = nullptr;
        error_    = error;
        finished_ = true;
        done_.notify_one();
    }
}

void PeerExecutor::run(const std::function<void()>& peer, const std::function<void()>& local) {
    {
        std::lock_guard lock(mutex_);
        task_     = &peer;
        finished_ = false;
        error_    = nullptr;
    }
    wake_.notify_one();
    std::exception_ptr local_error;
    try {
        local();
    } catch (...) {
        local_error = std::current_exception();
    }
    std::exception_ptr peer_error;
    {
        std::unique_lock lock(mutex_);
        done_.wait(lock, [this] { return finished_; });
        peer_error = error_;
    }
    if (local_error) { std::rethrow_exception(local_error); }
    if (peer_error) { std::rethrow_exception(peer_error); }
}

} // namespace ninfer::models::qwen3_5::detail
