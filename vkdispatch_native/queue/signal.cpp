#include <chrono>

#include "../base.hh"
#include "signal.hh"

#include "../context/context.hh"

#define NULL_TIMESTAMP ((uint64_t)0xFFFFFFFFFFFFFFFF)

Signal::Signal(struct Context* context) : state(false) {
    this->ctx = context;
    this->timestamp = NULL_TIMESTAMP;
    this->timestamp_queue_index = -1;
}

/*
* This function sets the state of the signal to true, indicating that the condition has occurred.
*/
void Signal::notify(int queue_index, uint64_t timestamp) {
    std::unique_lock<std::mutex> lock(mutex);
    this->timestamp = timestamp;
    this->timestamp_queue_index = queue_index;
    state.store(true, std::memory_order_release);
    cv.notify_all();
}

/*
* This function sets the state of the signal to false, indicating that the condition has not occurred.
* It allows threads to wait for the signal again.
*/
void Signal::reset() {
    std::unique_lock<std::mutex> lock(mutex);
    state.store(false, std::memory_order_release);
}

bool Signal::try_host_wait() {
    std::unique_lock<std::mutex> lock(mutex);
    
    bool notified = cv.wait_for(lock, std::chrono::seconds(1), [this] {
        LOG_VERBOSE("Checking signal");

        if(ctx->running.load(std::memory_order_acquire) == false) {
            set_error("Context is not running, cannot wait for signal");
            return true;
        }
        
        return state.load(std::memory_order_acquire);
    });

    return notified;
}

bool Signal::try_device_wait(int queue_index) {
    if(this->timestamp == NULL_TIMESTAMP) {
        set_error("Signal timestamp is NULL, cannot wait for device");
        return false;
    }

    if(queue_index < 0 || queue_index >= ctx->queues.size()) {
        set_error("Invalid queue index %d for device wait", queue_index);
        return false;
    }

    return ctx->queues[queue_index]->try_wait_for_timestamp(timestamp);
}

/*
* This function blocks the calling thread until the signal is notified.
*/
bool Signal::try_wait(bool wait_for_timestamp, int queue_index) {
    LOG_VERBOSE("Trying to wait on signal %p (wait_for_timestamp=%d, queue_index=%d)...", this, wait_for_timestamp, queue_index);

    if (state.load(std::memory_order_acquire)) {
        LOG_VERBOSE("Signal %p already notified", this);

        if (!wait_for_timestamp) {
            LOG_VERBOSE("No need to wait for timestamp, returning");
            return true;
        }

        LOG_VERBOSE("Waiting for timestamp %llu on queue %d", this->timestamp, queue_index);

        return try_device_wait(queue_index);
    }

    LOG_VERBOSE("Waiting for host notification on signal %p...", this);
    if(!try_host_wait()) {
        LOG_VERBOSE("Host wait for signal %p timed out", this);
        return false;
    }

    if(!wait_for_timestamp) {
        LOG_VERBOSE("No need to wait for timestamp, returning");
        return true;
    }

    LOG_VERBOSE("Waiting for timestamp %llu on queue %d", this->timestamp, queue_index);
    return try_device_wait(queue_index);
}