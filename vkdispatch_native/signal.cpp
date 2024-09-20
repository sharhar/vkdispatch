#include "internal.h"

#include <algorithm>

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

Semaphore::Semaphore() {
    pendingSubmissions = 0;
    waiting = false;
}

void Semaphore::addJob() {
    std::unique_lock<std::mutex> lock(mtx);
    pendingSubmissions = pendingSubmissions + 1;
}

void Semaphore::submitJob(VkDevice device, struct MyFence fence) {
    std::unique_lock<std::mutex> lock(mtx);
    if(fence.fence != VK_NULL_HANDLE)
        this->fences.push_back(fence);

    int current_index = 0;
    size_t fence_count = fences.size();
    while(current_index < fence_count) {
        if(fences[current_index].device != device) {
            current_index++;
            continue;
        }

        VkResult result = vkGetFenceStatus(fences[current_index].device, fences[current_index].fence);
        if(result == VK_SUCCESS) {
            fences.erase(fences.begin() + current_index);
            fence_count = fences.size();
        } else {
            current_index++;
        }
    }

    pendingSubmissions = pendingSubmissions - 1;
    
    if (pendingSubmissions == 0)
        cv.notify_all();
}

void Semaphore::finishJob(struct MyFence fence, std::function<void()> callback) {
    std::unique_lock<std::mutex> lock(mtx);
    callback();
    
    if(!waiting) {
        int fenceIndex = -1;

        for(int i = 0; i < fences.size(); i++) {
            if(fences[i].fence == fence.fence && fences[i].device == fence.device) {
                fenceIndex = i;
                break;
            }
        }

        if(fenceIndex != -1) {
            fences.erase(fences.begin() + fenceIndex);
        }
    }
}

void Semaphore::waitForIdle() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this]() { return pendingSubmissions == 0; });

    waiting = true;

    for(int i = 0; i < fences.size(); i++) {
        vkWaitForFences(fences[i].device, 1, &fences[i].fence, VK_TRUE, UINT64_MAX);
    }

    waiting = false;

    fences.clear();
}