#include "internal.h"

struct ComputePlan* stage_compute_plan_create_extern(struct Context* ctx, struct ComputePlanCreateInfo* create_info) {
    struct ComputePlan* plan = new struct ComputePlan();
    plan->ctx = ctx;
    plan->pc_size = create_info->pc_size;

    plan->pipelineLayouts = new VKLPipelineLayout*[ctx->deviceCount];
    plan->pipelines = new VKLPipeline*[ctx->deviceCount];
    plan->descriptorSets = new VKLDescriptorSet*[ctx->deviceCount];

    for (int i = 0; i < ctx->deviceCount; i++) {
        VKLPipelineLayoutCreateInfo layoutCreateInfo = VKLPipelineLayoutCreateInfo();
        layoutCreateInfo.device(ctx->devices[i]);

        layoutCreateInfo.addShaderModuleSource(create_info->shader_source, VK_SHADER_STAGE_COMPUTE_BIT, "main", "compute_shader");

        if (create_info->pc_size > 0)
            layoutCreateInfo.addPushConstant(VK_SHADER_STAGE_COMPUTE_BIT, 0, create_info->pc_size);

        if(create_info->binding_count > 0) {
            VKLDescriptorSetLayoutCreateInfo& descriptorSetLayoutCreateInfo = layoutCreateInfo.addDescriptorSet();

            for (int j = 0; j < create_info->binding_count; j++) {
                if(create_info->descriptorTypes[j] != DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                    LOG_ERROR("Only storage buffers are supported for now");
                    return NULL;
                }

                descriptorSetLayoutCreateInfo.addBinding(j, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
            }

            descriptorSetLayoutCreateInfo.end();
        }
                
        plan->pipelineLayouts[i] = new VKLPipelineLayout(layoutCreateInfo);

        plan->pipelines[i] = new VKLPipeline();
        plan->pipelines[i]->create(VKLPipelineCreateInfo().layout(plan->pipelineLayouts[i]));

        if(create_info->binding_count > 0) {
            plan->descriptorSets[i] = new VKLDescriptorSet(plan->pipelineLayouts[i], 0);
        } else {
            plan->descriptorSets[i] = NULL;
        }
    }

    return plan;
}

void stage_compute_bind_extern(struct ComputePlan* plan, unsigned int binding, void* object) {
    struct Buffer* buffer = (struct Buffer*)object;

    for (int i = 0; i < plan->ctx->deviceCount; i++) {
        plan->descriptorSets[i]->writeBuffer(binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buffer->buffers[i], 0, VK_WHOLE_SIZE);
    }
}

struct ComputeRecordInfo {
    struct ComputePlan* plan;
    unsigned int blocks_x;
    unsigned int blocks_y;
    unsigned int blocks_z;
    unsigned int pc_size;
};

void stage_compute_record_extern(struct CommandList* command_list, struct ComputePlan* plan, unsigned int blocks_x, unsigned int blocks_y, unsigned int blocks_z) {
    struct ComputeRecordInfo* my_compute_info = (struct ComputeRecordInfo*)malloc(sizeof(struct ComputeRecordInfo));
    my_compute_info->plan = plan;
    my_compute_info->blocks_x = blocks_x;
    my_compute_info->blocks_y = blocks_y;
    my_compute_info->blocks_z = blocks_z;
    my_compute_info->pc_size = plan->pc_size;

    command_list->stages.push_back({
        [](VKLCommandBuffer* cmd_buffer, struct Stage* stage, void* instance_data, int device) {
            LOG_INFO("Executing Compute");

            struct ComputeRecordInfo* my_compute_info = (struct ComputeRecordInfo*)stage->user_data;

            cmd_buffer->bindPipeline(my_compute_info->plan->pipelines[device]);

            if(my_compute_info->plan->descriptorSets[device] != NULL)
                cmd_buffer->bindDescriptorSet(my_compute_info->plan->descriptorSets[device]);
            
            if(my_compute_info->pc_size > 0)
                cmd_buffer->pushConstants(my_compute_info->plan->pipelines[device], VK_SHADER_STAGE_COMPUTE_BIT, 0, my_compute_info->pc_size, instance_data);

            cmd_buffer->dispatch(my_compute_info->blocks_x, my_compute_info->blocks_y, my_compute_info->blocks_z);
        },
        my_compute_info,
        plan->pc_size
    });
}