#include <chrono>

#include "../base.hh"
#include "signal.hh"

#include "../context/context.hh"


Signal::Signal(struct Context* context) : state(false) {
    this->ctx = context;
}

/*
* This function sets the state of the signal to true, indicating that the condition has occurred.
*/
void Signal::notify() {
    std::unique_lock<std::mutex> lock(mutex);
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

/*
* This function blocks the calling thread until the signal is notified.
*/
void Signal::wait() {
    if (state.load(std::memory_order_acquire)) {
        return; // If the signal is already notified, return immediately
    }

    std::unique_lock<std::mutex> lock(mutex);
    
    while(true) {
        bool ready = cv.wait_for(lock, std::chrono::seconds(1), [this] {
            LOG_VERBOSE("Checking signal");

            if(ctx->running.load(std::memory_order_acquire) == false) {
                set_error("Context is not running, cannot wait for signal");
                return true;
            }
            
            return state.load(std::memory_order_acquire);
        });

        if (ready) {
            return;
        }

        LOG_VERBOSE("Timeout expired, rechecking...");
    }
}