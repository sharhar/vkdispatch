#include "internal.h"

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

void Queue::push(struct WorkInfo* elem) {
    std::unique_lock<std::mutex> lock(this->mutex);

    LOG_INFO("Pushing work info to queue");

    this->cv_pop.wait(lock, [this] () {
        LOG_INFO("Checking for space to push to %d < %d ?", this->data.size(), this->max_size);
        return this->data.size() < this->max_size;
    });

    LOG_INFO("Found space, pushing to queue");
    this->data.push_back(elem);
    LOG_INFO("Notifying all");
    this->cv_push.notify_all();
    LOG_INFO("finished push");
}

bool Queue::pop(struct WorkInfo** elem, std::function<bool(struct WorkInfo* arg)> check) {
    std::unique_lock<std::mutex> lock(this->mutex);

    LOG_INFO("Popping work info from queue");

    int found_index = -1;
    this->cv_push.wait(lock, [this, check, &found_index] () {
        LOG_INFO("Checking for work info in queue");

        if(!running)
            return true;

        if(this->data.size() == 0)
            return false;

        LOG_INFO("Work queue size: %d", this->data.size());
        
        for(int i = 0; i < this->data.size(); i++) {
            if(check(this->data[i])) {
                LOG_INFO("Found work info in queue");
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

    LOG_INFO("Removing index %d from work queue", found_index);

    *elem = this->data[found_index];
    this->data.erase(this->data.begin() + found_index);

    LOG_INFO("Notifying all of pop");
    this->cv_pop.notify_all();


    return get_error_string_extern() == NULL;
}