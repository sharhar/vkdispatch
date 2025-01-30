#include "../include/internal.hh"

static size_t __work_id = 0;

WorkQueue::WorkQueue(int max_work_items, int max_programs) {
    work_infos = new WorkInfo2[max_work_items];
    program_infos = new ProgramInfo[max_programs];
    work_info_count = max_work_items;
    program_info_count = max_programs;
    running = true;

    for(int i = 0; i < max_work_items; i++) {
        work_infos[i].dirty = false;
        work_infos[i].header = (struct WorkHeader*)malloc(sizeof(struct WorkHeader) + 16 * 1024);
        memset(work_infos[i].header, 0, sizeof(struct WorkHeader) + 16 * 1024);
        work_infos[i].header->array_size = 16 * 1024;
        work_infos[i].header->info_index = i;
    }

    for(int i = 0; i < max_programs; i++) {
        program_infos[i].ref_count = 0;
        program_infos[i].header = (struct ProgramHeader*)malloc(sizeof(struct ProgramHeader) + 16 * 1024);
        memset(program_infos[i].header, 0, sizeof(struct ProgramHeader) + 16 * 1024);
        program_infos[i].header->array_size = 16 * 1024;
        program_infos[i].header->info_index = i;
    }
}

void WorkQueue::stop() {
    std::unique_lock<std::mutex> lock(this->mutex);
    this->running = false;
    this->cv_push.notify_all();
}

void WorkQueue::push(struct CommandList* command_list, void* instance_buffer, unsigned int instance_count, int stream_index, Signal* signal) {
    std::unique_lock<std::mutex> lock(this->mutex);
    
    auto start = std::chrono::high_resolution_clock::now();

    int found_indicies[2] = {-1, -1};

    this->cv_pop.wait(lock, [this, start, command_list, &found_indicies] () {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        if(elapsed.count() > 5) {
            set_error("Timed out waiting for room in queue");
            return true;
        }

        int program_index = -1;

        for(int i = 0; i < this->program_info_count; i++) {
            if(this->program_infos[i].ref_count < 0) {
                set_error("Program reference count is negative!!!!");
                return true;
            }

            if(this->program_infos[i].program_id == command_list->program_id) {
                program_index = i;
                break;
            }

            if(this->program_infos[i].ref_count == 0) {
                program_index = i;
            }
        }

        if(program_index == -1) {
            return false;
        }

        int work_index = -1;

        for(int i = 0; i < this->work_info_count; i++) {
            if(!this->work_infos[i].dirty) {
                work_index = i;
                break;
            }
        }

        if(work_index == -1) {
            return false;
        }

        found_indicies[0] = program_index;
        found_indicies[1] = work_index;

        return true;
    });

    RETURN_ON_ERROR(;)

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if(elapsed.count() >= 5) {
        return;
    }

    work_infos[found_indicies[1]].program_index = found_indicies[0];
    work_infos[found_indicies[1]].stream_index = stream_index;
    work_infos[found_indicies[1]].dirty = true;
    work_infos[found_indicies[1]].state = WORK_STATE_PENDING;
    work_infos[found_indicies[1]].work_id = __work_id;
    __work_id += 1;

    struct ProgramHeader* program_header = this->program_infos[found_indicies[0]].header;
    struct WorkHeader* work_header = this->work_infos[found_indicies[1]].header;

    if(this->program_infos[found_indicies[0]].program_id != command_list->program_id) {
        if(this->program_infos[found_indicies[0]].ref_count != 0) {
            set_error("Program ID mismatch!!");
            return;
        }

        size_t program_size = command_list->commands.size() * sizeof(struct CommandInfo);

        if(program_size > program_header->array_size) {
            program_header = (struct ProgramHeader*)realloc(program_header, sizeof(struct ProgramHeader) + program_size);
            program_header->array_size = program_size;
            program_header->info_index = found_indicies[0];
            this->program_infos[found_indicies[0]].header = program_header;
        }

        memcpy(&program_header[1], command_list->commands.data(), program_size);
        program_header->command_count = command_list->commands.size();
        program_header->conditional_boolean_count = command_list->conditional_boolean_count;
        this->program_infos[found_indicies[0]].program_id = command_list->program_id;
    }

    size_t work_size = command_list_get_instance_size_extern(command_list) * instance_count;

    if(work_size > work_header->array_size) {
        work_header = (struct WorkHeader*)realloc(work_header, sizeof(struct WorkHeader) + work_size);
        work_header->array_size = work_size;
        work_header->info_index = found_indicies[1];
        this->work_infos[found_indicies[1]].header = work_header;
    }

    work_header->instance_count = instance_count;
    work_header->instance_size = command_list_get_instance_size_extern(command_list);
    work_header->signal = signal;
    work_header->program_header = program_header;
    
    if(work_size > 0)
        memcpy(&work_header[1], instance_buffer, work_size);
    
    this->program_infos[found_indicies[0]].ref_count += 1;

    this->cv_push.notify_all();
}

bool WorkQueue::pop(struct WorkHeader** header, int stream_index) {
    std::unique_lock<std::mutex> lock(this->mutex);

    this->cv_push.wait(lock, [this, stream_index, header] () {
        if(!running) {
            return true;
        }

        int selected_index = -1;
        size_t work_id = ~((size_t)0);

        for(int i = 0; i < this->work_info_count; i++) {
            if(this->work_infos[i].dirty &&
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

    return running;
}

void WorkQueue::finish(struct WorkHeader* header) {
    std::unique_lock<std::mutex> lock(this->mutex);
    program_infos[header->program_header->info_index].ref_count -= 1;
    work_infos[header->info_index].dirty = false;
    this->cv_pop.notify_all();
}