#include "handles.hh"

#include "context.hh"

HandleManager::HandleManager(Context* ctx) {
    next_handle = 1;
    queue_count = ctx->queues.size();

    queue_to_device_map = new int[queue_count];
    for (int i = 0; i < queue_count; i++) {
        queue_to_device_map[i] = ctx->queues[i]->device_index;
    }
}

uint64_t HandleManager::register_device_handle(const char* name) {
    return register_handle(name, queue_count, true);
}

uint64_t HandleManager::register_queue_handle(const char* name) {
    return register_handle(name, queue_count, false);
}

void atomic_max_func(std::atomic<uint64_t>* atom, uint64_t val) {
    uint64_t current_val = atom->load(std::memory_order_relaxed);
    while (val > current_val) {
        if (atom->compare_exchange_weak(current_val, val, std::memory_order_relaxed)) {
            break;
        }
    }
}

uint64_t HandleManager::register_handle(const char* name, size_t count, bool per_device) {
    std::unique_lock lock(handle_mutex);

    if(per_device && count != queue_count) {
        LOG_ERROR("Per device handle count does not match queue count");
        return 0;
    }
    
    uint64_t handle = next_handle++;
    uint64_t* handle_data = new uint64_t[count];
    for (int i = 0; i < count; i++) {
        handle_data[i] = 0;
    }

    std::atomic<uint64_t>* timestamps = new std::atomic<uint64_t>[count];
    for (int i = 0; i < count; i++) {
        timestamps[i] = 0;
    }

    struct HandleHeader header;
    header.handle = handle;
    header.count = count;
    header.delete_count = 0;
    header.per_device = per_device;
    header.data = handle_data;
    header.timestamps = timestamps;
    header.name = name;

    handles[handle] = header;

    return handle;
}

void HandleManager::set_handle(int64_t index, uint64_t handle, uint64_t value) {
    std::unique_lock lock(handle_mutex);

    if(handles[handle].per_device) {
        LOG_ERROR("Handle is per device");
        return;
    }

    if(index >= handles[handle].count || index < 0) {
        LOG_ERROR("Index %d out of bounds for handle %s (%d)", index, handles[handle].name, handle);
        return;
    }

    handles[handle].data[index] = value;
}

void HandleManager::set_handle_per_device(int device_index, uint64_t handle, std::function<uint64_t(int)> value_func) {
    std::unique_lock lock(handle_mutex);

    if(!handles[handle].per_device) {
        LOG_ERROR("Handle is not per device");
        return;
    }

    bool found_any = false;
    bool found_all = true;

    for (int i = 0; i < queue_count; i++) {
        if (queue_to_device_map[i] == device_index) {
            found_all = found_all && (handles[handle].data[i] != 0);
            found_any = found_any || (handles[handle].data[i] != 0);
        }
    }

    if(found_any && !found_all) {
        LOG_ERROR("Handle already set for some queues but not all");
        return;
    }

    if(!found_any && found_all) {
        LOG_ERROR("Some weird stuff is going on");
        return;
    }

    if(found_all && found_any) {
        return;
    }

    uint64_t value = value_func(device_index);

    for (int i = 0; i < queue_count; i++) {
        if (queue_to_device_map[i] == device_index) {
            handles[handle].data[i] = value;
        }
    }
}

uint64_t HandleManager::get_handle(int64_t index, uint64_t handle, uint64_t timestamp) {
    std::shared_lock lock(handle_mutex);

    if (timestamp != 0)
        atomic_max_func(&handles[handle].timestamps[index], timestamp);

    return handles[handle].data[index];
}

uint64_t* HandleManager::get_handle_pointer(int64_t index, uint64_t handle, uint64_t timestamp) {
    std::shared_lock lock(handle_mutex);

    if (timestamp != 0)
        atomic_max_func(&handles[handle].timestamps[index], timestamp);

    return &handles[handle].data[index];
}

uint64_t HandleManager::get_handle_no_lock(int64_t index, uint64_t handle) {
    return handles[handle].data[index];
}

uint64_t* HandleManager::get_handle_pointer_no_lock(int64_t index, uint64_t handle) {
    return &handles[handle].data[index];
}

uint64_t HandleManager::get_handle_timestamp(int64_t index, uint64_t handle) {
    std::shared_lock lock(handle_mutex);
    return handles[handle].timestamps[index].load(std::memory_order_relaxed);
}

void HandleManager::destroy_handle(int64_t index, uint64_t handle) {
    std::unique_lock lock(handle_mutex);

    handles[handle].delete_count++;
    handles[handle].data[index] = 0;

    if (handles[handle].delete_count >= handles[handle].count) {
        delete[] handles[handle].data;
        delete[] handles[handle].timestamps;
        handles.erase(handle);
    }
}

void HandleManager::destroy_handle_per_device(int device_index, uint64_t handle, std::function<void(uint64_t, uint64_t)> destroy_func) {
    std::unique_lock lock(handle_mutex);

    if(!handles[handle].per_device) {
        LOG_ERROR("Handle is not per device");
        return;
    }

    bool found_any = false;
    bool found_all = true;

    for (int i = 0; i < queue_count; i++) {
        if (queue_to_device_map[i] == device_index) {
            found_all = found_all && (handles[handle].data[i] != 0);
            found_any = found_any || (handles[handle].data[i] != 0);
        }
    }

    if(found_any && !found_all) {
        LOG_ERROR("Handle already set for some queues but not all");
        return;
    }

    if(!found_any && found_all) {
        LOG_ERROR("Some weird stuff is going on");
        return;
    }

    if(found_all && found_any) {
        return;
    }

    uint64_t handle_value = 0;
    uint64_t handle_timestamp = 0;

    for (int i = 0; i < queue_count; i++) {
        if (queue_to_device_map[i] == device_index) {
            
            if (handle_value == 0) {
                handle_value = handles[handle].data[i];
            } else if (handles[handle].data[i] != handle_value){
                LOG_ERROR("Handle value mismatch for handle %s (%d) at index %d: expected %llu, got %llu", handles[handle].name, handle, i, handle_value, handles[handle].data[i]);
                return;
            }

            uint64_t current_timestamp = handles[handle].timestamps[i].load(std::memory_order_relaxed);
            
            if (handle_timestamp < current_timestamp) {
                handle_timestamp = current_timestamp;
            }
        }
    }

    if (handle_value == 0) {
        LOG_ERROR("Handle value is 0 for handle %s (%d)", handles[handle].name, handle);
        return;
    }

    destroy_func(handle_value, handle_timestamp);
}