#include "internal.h"

static size_t __work_id = 0;

WorkQueue::WorkQueue(int max_work_items, int max_programs) {
    work_infos = new WorkInfo2[max_work_items];
    work_info_count = max_work_items;
    running = true;

    for(int i = 0; i < max_work_items; i++) {
        work_infos[i].dirty = false;
        work_infos[i].header = (struct WorkHeader*)malloc(sizeof(struct WorkHeader) + 16 * 1024);
        memset(work_infos[i].header, 0, sizeof(struct WorkHeader) + 16 * 1024);
        work_infos[i].header->array_size = 16 * 1024;
        work_infos[i].header->info_index = i;
    }
}

void WorkQueue::stop() {
    std::unique_lock<std::mutex> lock(this->mutex);
    this->running = false;
    this->cv_push.notify_all();
}

void WorkQueue::waitIdle() {
    /*
    std::unique_lock<std::mutex> lock(this->mutex);
    this->cv_pop.wait(lock, [this] () {
        for(int i = 0; i < this->work_info_count; i++) {
            if(this->work_infos[i].dirty == true) {
                return false;
            }
        }

        return true;
    });
    */
}

void WorkQueue::push(struct CommandList* command_list, void* instance_buffer, unsigned int instance_count, int stream_index, Signal* signal) {
    std::unique_lock<std::mutex> lock(this->mutex);
    
    auto start = std::chrono::high_resolution_clock::now();

    int found_index[1] = {-1};

    this->cv_pop.wait(lock, [this, start, &found_index] () {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        if(elapsed.count() > 5) {
            set_error("Timed out waiting for room in queue");
            return true;
        }

        int work_index = -1;

        for(int i = 0; i < this->work_info_count; i++) {
            if(this->work_infos[i].dirty == false) {
                work_index = i;
                break;
            }
        }

        if(work_index == -1) {
            return false;
        }

        found_index[0] = work_index;

        return true;
    });

    RETURN_ON_ERROR(;)

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if(elapsed.count() >= 5) {
        return;
    }

    work_infos[found_index[0]].stream_index = stream_index;
    work_infos[found_index[0]].dirty = true;
    work_infos[found_index[0]].state = WORK_STATE_PENDING;
    work_infos[found_index[0]].work_id = __work_id;
    __work_id += 1;

    struct WorkHeader* work_header = this->work_infos[found_index[0]].header;
    size_t work_size = command_list->instance_size * instance_count;

    if(work_size > work_header->array_size) {
        work_header = (struct WorkHeader*)realloc(work_header, sizeof(struct WorkHeader) + work_size);
        work_header->array_size = work_size;
        work_header->info_index = found_index[0];
        this->work_infos[found_index[0]].header = work_header;
    }

    work_header->command_list = command_list;
    work_header->instance_count = instance_count;
    work_header->instance_size = command_list->instance_size;
    work_header->signal = signal;
    
    if(work_size > 0)
        memcpy(&work_header[1], instance_buffer, work_size);

    this->cv_push.notify_all();
}

bool WorkQueue::pop(struct WorkHeader** header, int stream_index) {
    std::unique_lock<std::mutex> lock(this->mutex);
    
    //auto start = std::chrono::high_resolution_clock::now();

    this->cv_push.wait(lock, [this, stream_index, header] () {
        //auto end = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double> elapsed = end - start;

        if(!running) {
            return true;
        }
        
        //if(elapsed.count() > 5) {
        //    LOG_ERROR("Timed out waiting for work in queue");
        //    return true;
        //}

        int selected_index = -1;
        size_t work_id = ~((size_t)0);

        for(int i = 0; i < this->work_info_count; i++) {
            if(this->work_infos[i].dirty == true &&
               this->work_infos[i].state == WORK_STATE_PENDING &&
               this->work_infos[i].work_id < work_id &&
               (this->work_infos[i].stream_index == stream_index ||
                this->work_infos[i].stream_index == -1))
            {
                work_id = this->work_infos[i].work_id;
                selected_index = i;
            }
        }

        if(selected_index == -1) {
            return false;
        }

        *header = this->work_infos[selected_index].header;
        this->work_infos[selected_index].state = WORK_STATE_ACTIVE;

        return true;
    });

    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> elapsed = end - start;

    //if(elapsed.count() >= 5) {
    //    return false;
    //}

    return running;
}

void WorkQueue::finish(struct WorkHeader* header) {
    //program_infos[header->program_header->info_index].ref_count.fetch_sub(1, std::memory_order_relaxed);
    work_infos[header->info_index].dirty = false;
    this->cv_pop.notify_all();
}