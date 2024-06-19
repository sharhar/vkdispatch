#include "internal.h"

#include <chrono>

Queue::Queue(int max_size) {
    this->max_size = max_size;
    this->data.reserve(max_size);
    this->running = true;
}
void Queue::stop() {
    std::unique_lock<std::mutex> lock(this->mutex);
    this->running = false;
    this->cv_push.notify_all();
}

void Queue::push(struct WorkInfo elem) {
    std::unique_lock<std::mutex> lock(this->mutex);
    
    auto start = std::chrono::high_resolution_clock::now();

    this->cv_pop.wait(lock, [this, start] () {
        LOG_VERBOSE("Checking push");

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        if(elapsed.count() > 5) {
            LOG_ERROR("Timed out waiting for room in queue");
            return true;
        }

        LOG_VERBOSE("Checking for room");
        return this->data.size() < this->max_size;
    });

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if(elapsed.count() < 5) {
        this->data.push_back(elem);
        this->cv_push.notify_all();
    }
}

bool Queue::pop(struct WorkInfo* elem, std::function<bool(const struct WorkInfo& arg)> check, std::function<void(const struct WorkInfo& arg)> finalize) {
    std::unique_lock<std::mutex> lock(this->mutex);

    int found_index = -1;
    this->cv_push.wait(lock, [this, check, &found_index] () {
        LOG_VERBOSE("Checking pop");

        if(!running)
            return true;

        if(this->data.size() == 0)
            return false;
        
        for(int i = 0; i < this->data.size(); i++) {
            if(check(this->data[i])) {
                found_index = i;
                return true;
            }
        }

        return false;
    });

    if(found_index == -1) {
        set_error("Could not find element in queue");
        return false;
    }
    
    if(!running)
        return false;

    *elem = this->data[found_index];

    finalize(this->data[found_index]);

    this->data.erase(this->data.begin() + found_index);
    this->cv_pop.notify_all();

    return get_error_string_extern() == NULL;
}