#include "internal.h"

Signal::Signal(std::shared_ptr<Signal> parent) : parent(parent), state(false) {}

void Signal::notify() {
    if(parent != nullptr) {
        parent->wait();
    }

    std::unique_lock<std::mutex> lock(mutex);
    state = true;
    cv.notify_all();
}

void Signal::wait() {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this] { return state; });
}