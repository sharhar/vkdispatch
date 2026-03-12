#include <chrono>
#include <cstring>

#include "work_queue.hh"
#include "../objects/command_list.hh"
#include "../objects/objects_extern.hh"


static size_t __work_id = 0;

WorkQueue::WorkQueue(int max_work_items, int max_programs) {
    work_infos = new WorkInfo[max_work_items];
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
        work_infos[i].header->name = nullptr;
    }

    for(int i = 0; i < max_programs; i++) {
        program_infos[i].ref_count = 0;
        program_infos[i].commands = std::make_shared<std::vector<struct CommandInfo>>();
        program_infos[i].program_id = 0;
    }
}

void WorkQueue::stop() {
    std::unique_lock<std::mutex> lock(this->mutex);
    this->running = false;
    this->cv_push.notify_all();
}

int WorkQueue::get_program_index(struct CommandList* command_list) {
    int program_index = -1;

    for(int i = 0; i < this->program_info_count; i++) {
        // Sanity check
        if(this->program_infos[i].ref_count < 0) {
            set_error("Program reference count (%d) is negative!", this->program_infos[i].ref_count);
            return -2;
        }

        // Program already exists, return its index
        if(this->program_infos[i].program_id == command_list->program_id) {
            return i;
        }

        // Found an available slot
        if(this->program_infos[i].ref_count == 0) {
            program_index = i;
        }
    }

    return program_index;
}

int WorkQueue::get_work_index() {
    for(int i = 0; i < this->work_info_count; i++) {
        if(!this->work_infos[i].dirty) {
            return i;
        }
    }

    return -1;
}

void WorkQueue::prepare_work(int work_index, int program_index, struct CommandList* command_list, void* instance_buffer, unsigned int instance_count, int queue_index, int record_type, const char* name) {
    // Setup work info
    work_infos[work_index].program_index = program_index;
    work_infos[work_index].queue_index = queue_index;
    work_infos[work_index].dirty = true;
    work_infos[work_index].state = WORK_STATE_PENDING;
    work_infos[work_index].work_id = __work_id;
    __work_id += 1;

    struct WorkHeader* work_header = this->work_infos[work_index].header;

    // Update the program if needed
    if(this->program_infos[program_index].program_id != command_list->program_id) {
        // Sanity check
        if(this->program_infos[program_index].ref_count != 0) {
            set_error("Program ID mismatch!!");
            return;
        }

        // Update program commands
        this->program_infos[program_index].commands->clear();
        for(CommandInfo command : command_list->commands) {
            this->program_infos[program_index].commands->push_back(command);
        }

        // Update program ID
        this->program_infos[program_index].program_id = command_list->program_id;
    }

    size_t work_size = command_list_get_instance_size_extern(command_list) * instance_count;

    // Resize work header if needed
    if(work_size > work_header->array_size) {
        work_header = (struct WorkHeader*)realloc(work_header, sizeof(struct WorkHeader) + work_size);
        work_header->array_size = work_size;
        work_header->info_index = work_index;
        this->work_infos[work_index].header = work_header;
    }

    // Setup work header
    work_header->instance_count = instance_count;
    work_header->instance_size = command_list_get_instance_size_extern(command_list);
    work_header->commands = this->program_infos[program_index].commands;
    work_header->program_info_index = program_index;
    work_header->record_type = (RecordType)record_type;
    work_header->name = name;
    
    // Copy instance data if needed
    if(work_size > 0)
        memcpy(&work_header[1], instance_buffer, work_size);
    
    // Increment program reference count
    this->program_infos[program_index].ref_count += 1;
}

bool WorkQueue::push(struct CommandList* command_list, void* instance_buffer, unsigned int instance_count, int queue_index, int record_type, const char* name) {
    std::unique_lock<std::mutex> lock(this->mutex);

    int found_indicies[2] = {-1, -1};

    bool ready = this->cv_pop.wait_for(lock, std::chrono::seconds(1), [this, command_list, &found_indicies] () {
        if(!running) {
            return true;
        }

        int program_index = get_program_index(command_list);

        // Error occurred, return now and exit
        if(program_index == -2)
            return true;
        
        // No available program slots, try again later
        if(program_index == -1)
            return false;

        int work_index = get_work_index();

        // No available work slots, try again later
        if(work_index == -1)
            return false;

        found_indicies[0] = program_index;
        found_indicies[1] = work_index;

        return true;
    });

    if(!ready)
        return false;

    if(!running) {
        return true;
    }

    RETURN_ON_ERROR(true)

    prepare_work(found_indicies[1], found_indicies[0], command_list, instance_buffer, instance_count, queue_index, record_type, name);

    this->cv_push.notify_all();

    return true;
}

bool WorkQueue::pop(struct WorkHeader** header, int queue_index) {
    std::unique_lock<std::mutex> lock(this->mutex);

    LOG_VERBOSE("Waiting for work item for queue %d", queue_index);

    this->cv_push.wait(lock, [this, queue_index, header] () {
        LOG_VERBOSE("IN CV: running = %d, work_info_count = %d, program_info_count = %d", running, work_info_count, program_info_count);
        if(!running) {
            return true;
        }

        int selected_index = -1;
        size_t work_id = ~((size_t)0);

        for(int i = 0; i < this->work_info_count; i++) {
            if(this->work_infos[i].dirty &&
               this->work_infos[i].state == WORK_STATE_PENDING &&
               this->work_infos[i].work_id < work_id &&
               (this->work_infos[i].queue_index == queue_index ||
                this->work_infos[i].queue_index == -1))
            {
                work_id = this->work_infos[i].work_id;
                selected_index = i;
            }
        }

        LOG_VERBOSE("Selected index: %d", selected_index);

        if(selected_index == -1) {
            return false;
        }

        *header = this->work_infos[selected_index].header;
        this->work_infos[selected_index].state = WORK_STATE_ACTIVE;

        LOG_VERBOSE("Returning work header %p for queue %d", *header, queue_index);

        return true;
    });

    return running;
}

void WorkQueue::finish(struct WorkHeader* header) {
    std::unique_lock<std::mutex> lock(this->mutex);
    program_infos[header->program_info_index].ref_count -= 1;
    work_infos[header->info_index].dirty = false;
    this->cv_pop.notify_all();
}