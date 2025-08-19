#include "../base.hh"
#include "signal.hh"

Signal::Signal() : state(false) {
    
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
    
    auto start = std::chrono::high_resolution_clock::now();
    
    cv.wait(lock, [this, start] {
        LOG_VERBOSE("Checking signal");

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        if(elapsed.count() > 5) {
            set_error("Timed out waiting for signal");
            return true;
        }
        
        return state.load(std::memory_order_acquire);
    });
}