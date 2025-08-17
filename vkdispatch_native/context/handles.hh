#ifndef SRC_HANDLES_H_
#define SRC_HANDLES_H_

#include "../base.hh"

#include <functional>
#include <shared_mutex>
#include <unordered_map>

struct HandleHeader {
    uint64_t handle;
    size_t count;
    uint64_t* data;
    bool per_device;
    const char* name;
};

class HandleManager {
public:
    uint64_t next_handle;
    int queue_count;
    int* queue_to_device_map;
    std::shared_mutex handle_mutex;

    std::unordered_map<uint64_t, struct HandleHeader> handles;

    HandleManager(Context* ctx);

    uint64_t register_device_handle(const char* name);
    uint64_t register_queue_handle(const char* name);
    uint64_t register_handle(const char* name, size_t count, bool per_device);

    void set_handle(int64_t index, uint64_t handle, uint64_t value);
    void set_handle_per_device(int device_index, uint64_t handle, std::function<uint64_t(int)> value_func);
    uint64_t get_handle(int64_t index, uint64_t handle);
    uint64_t* get_handle_pointer(int64_t index, uint64_t handle);
    uint64_t get_handle_no_lock(int64_t index, uint64_t handle);
    uint64_t* get_handle_pointer_no_lock(int64_t index, uint64_t handle);
    void destroy_handle(int64_t index, uint64_t handle, std::function<void(uint64_t)> destroy_func);
};

#endif