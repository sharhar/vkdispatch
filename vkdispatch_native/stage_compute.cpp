#include "internal.h"

struct ComputePlan* stage_compute_plan_create_extern(struct Context* ctx, struct ComputePlanCreateInfo* create_info) {
    struct ComputePlan* plan = new struct ComputePlan();
    plan->ctx = ctx;
    plan->pc_size = create_info->pc_size;

    plan->pipelineLayouts = new VKLPipelineLayout*[ctx->deviceCount];
    plan->pipelines = new VKLPipeline*[ctx->deviceCount];

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

        plan->binding_count = create_info->binding_count;
    }

    return plan;
}

struct ComputeRecordInfo {
    struct ComputePlan* plan;
    struct DescriptorSet* descriptor_set;
    unsigned int blocks_x;
    unsigned int blocks_y;
    unsigned int blocks_z;
    unsigned int pc_size;
};

void stage_compute_record_extern(struct CommandList* command_list, struct ComputePlan* plan, struct DescriptorSet* descriptor_set, unsigned int blocks_x, unsigned int blocks_y, unsigned int blocks_z) {
    struct ComputeRecordInfo* my_compute_info = (struct ComputeRecordInfo*)malloc(sizeof(struct ComputeRecordInfo));
    my_compute_info->plan = plan;
    my_compute_info->descriptor_set = descriptor_set;
    my_compute_info->blocks_x = blocks_x;
    my_compute_info->blocks_y = blocks_y;
    my_compute_info->blocks_z = blocks_z;
    my_compute_info->pc_size = plan->pc_size;

    command_list->stages.push_back({
        [](VkCommandBuffer cmd_buffer, struct Stage* stage, void* instance_data, int device) {
            LOG_INFO("Executing Compute");

            struct ComputeRecordInfo* my_compute_info = (struct ComputeRecordInfo*)stage->user_data;

            vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, my_compute_info->plan->pipelines[device]->handle());

            if(my_compute_info->descriptor_set != NULL)
                vkCmdBindDescriptorSets(
                    cmd_buffer,
                    VK_PIPELINE_BIND_POINT_COMPUTE,
                    my_compute_info->plan->pipelineLayouts[device]->handle(),
                    0,
                    1,
                    &my_compute_info->descriptor_set->sets[device],
                    0,
                    NULL
                );

            if(my_compute_info->pc_size > 0)
                vkCmdPushConstants(
                    cmd_buffer, 
                    my_compute_info->plan->pipelineLayouts[device]->handle(),
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    0,
                    my_compute_info->pc_size,
                    instance_data
                );
            
            vkCmdDispatch(cmd_buffer, my_compute_info->blocks_x, my_compute_info->blocks_y, my_compute_info->blocks_z);
        },
        my_compute_info,
        plan->pc_size,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    });
}