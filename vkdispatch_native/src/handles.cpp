#include "../include/internal.hh"

HandleManager::HandleManager(Context* ctx) {
    next_handle = 1;
    stream_count = ctx->streams.size();

    stream_to_device_map = new int[stream_count];
    for (int i = 0; i < stream_count; i++) {
        stream_to_device_map[i] = ctx->streams[i]->device_index;
    }
}

uint64_t HandleManager::register_device_handle(const char* name) {
    return register_handle(name, stream_count, true);
}

uint64_t HandleManager::register_stream_handle(const char* name) {
    return register_handle(name, stream_count, false);
}

uint64_t HandleManager::register_handle(const char* name, size_t count, bool per_device) {
    std::unique_lock lock(handle_mutex);

    if(per_device && count != stream_count) {
        LOG_ERROR("Per device handle count does not match stream count");
        return 0;
    }
    
    uint64_t handle = next_handle++;
    uint64_t* handle_data = new uint64_t[count];
    for (int i = 0; i < count; i++) {
        handle_data[i] = 0;
    }

    struct HandleHeader header;
    header.handle = handle;
    header.count = count;
    header.per_device = per_device;
    header.data = handle_data;
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

    for (int i = 0; i < stream_count; i++) {
        if (stream_to_device_map[i] == device_index) {
            found_all = found_all && (handles[handle].data[i] != 0);
            found_any = found_any || (handles[handle].data[i] != 0);
        }
    }

    if(found_any && !found_all) {
        LOG_ERROR("Handle already set for some streams but not all");
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

    for (int i = 0; i < stream_count; i++) {
        if (stream_to_device_map[i] == device_index) {
            handles[handle].data[i] = value;
        }
    }
}

uint64_t HandleManager::get_handle(int64_t index, uint64_t handle) {
    std::shared_lock lock(handle_mutex);

    return handles[handle].data[index];
}

uint64_t* HandleManager::get_handle_pointer(int64_t index, uint64_t handle) {
    std::shared_lock lock(handle_mutex);

    return &handles[handle].data[index];
}

void HandleManager::destroy_handle(int64_t index, uint64_t handle, std::function<void(uint64_t)> destroy_func) {
    std::unique_lock lock(handle_mutex);

    //destroy_func((T)handles[handle].data[stream_index]);
    //delete (T)handles[handle].data[stream_index];
    //handles.erase(handle);
}