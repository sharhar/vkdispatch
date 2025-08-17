#ifndef SRC_COMMAND_LIST_H
#define SRC_COMMAND_LIST_H

#include "../base.hh"

#include <memory>
#include <functional>

#include "../queue/barrier_manager.hh"

struct CommandInfo {
    std::shared_ptr<std::function<void(VkCommandBuffer, int, int, int, void*, BarrierManager*)>> func;
    VkPipelineStageFlags pipeline_stage;
    size_t pc_size;
    const char* name;
};

struct CommandList {
    struct Context* ctx;
    std::vector<struct CommandInfo> commands;
    size_t compute_instance_size;
    size_t program_id;
};

void command_list_record_command(
    struct CommandList* command_list, 
    const char* name,
    size_t pc_size,
    VkPipelineStageFlags pipeline_stage,
    std::function<void(VkCommandBuffer, int, int, int, void*, BarrierManager*)> func
);


#endif // SRC_COMMAND_LIST_H