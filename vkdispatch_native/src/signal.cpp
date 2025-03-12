#include "../include/internal.hh"

Signal::Signal() : state(false) {}

/*
* This function sets the state of the signal to true, indicating that the condition has occurred.
*/
void Signal::notify() {
    std::unique_lock<std::mutex> lock(mutex);
    state = true;
    cv.notify_all();
}

/*
* This function blocks the calling thread until the signal is notified.
*/
void Signal::wait() {
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
        
        return state;
    });
}