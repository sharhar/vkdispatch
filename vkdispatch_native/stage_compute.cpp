#include "internal.h"

#include <glslang/Public/resource_limits_c.h>

uint32_t* util_compile_shader_code(glslang_stage_t stage, size_t* size, const char* shader_source, const char* shader_name) {
    glslang_input_t input = {};
	input.language = GLSLANG_SOURCE_GLSL;
	input.stage = stage;
	input.client = GLSLANG_CLIENT_VULKAN;
	input.client_version = GLSLANG_TARGET_VULKAN_1_2;
	input.target_language = GLSLANG_TARGET_SPV;
	input.target_language_version = GLSLANG_TARGET_SPV_1_3;
	input.code = shader_source;
	input.default_version = 100;
	input.default_profile = GLSLANG_NO_PROFILE;
	input.force_default_version_and_profile = false;
	input.forward_compatible = false;
	input.messages = GLSLANG_MSG_DEFAULT_BIT;
	input.resource = glslang_default_resource();

    glslang_shader_t* shader = glslang_shader_create(&input);

    if (!glslang_shader_preprocess(shader, &input))	{
        LOG_ERROR("GLSL preprocessing failed %s", shader_name);
        LOG_ERROR("%s", glslang_shader_get_info_log(shader));
        LOG_ERROR("%s", glslang_shader_get_info_debug_log(shader));
        LOG_ERROR("%s", input.code);
        glslang_shader_delete(shader);
        return NULL;
    }

    if (!glslang_shader_parse(shader, &input)) {
        LOG_ERROR("GLSL parsing failed %s", shader_name);
        LOG_ERROR("%s", glslang_shader_get_info_log(shader));
        LOG_ERROR("%s", glslang_shader_get_info_debug_log(shader));
        LOG_ERROR("%s", glslang_shader_get_preprocessed_code(shader));
        glslang_shader_delete(shader);
        return NULL;
    }

    glslang_program_t* program = glslang_program_create();
    glslang_program_add_shader(program, shader);

    if (!glslang_program_link(program, GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT)) {
        LOG_ERROR("GLSL linking failed %s", shader_name);
        LOG_ERROR("%s", glslang_program_get_info_log(program));
        LOG_ERROR("%s", glslang_program_get_info_debug_log(program));
        glslang_program_delete(program);
        glslang_shader_delete(shader);
        return NULL;
    }

    glslang_program_SPIRV_generate(program, stage);

    *size = glslang_program_SPIRV_get_size(program) * sizeof(uint32_t);

    uint32_t* words = (uint32_t*)malloc(*size);
    glslang_program_SPIRV_get(program, words);

    const char* spirv_messages = glslang_program_SPIRV_get_messages(program);
    if (spirv_messages)
        LOG_ERROR("(%s) %s\b", shader_name, spirv_messages);

    glslang_program_delete(program);
    glslang_shader_delete(shader);

    return words;
}

struct ComputePlan* stage_compute_plan_create_extern(struct Context* ctx, struct ComputePlanCreateInfo* create_info) {
    struct ComputePlan* plan = new struct ComputePlan();
    plan->ctx = ctx;
    plan->pc_size = create_info->pc_size;
    plan->binding_count = create_info->binding_count;
    plan->poolSizes.reserve(ctx->deviceCount);

    for (int i = 0; i < ctx->deviceCount; i++) {

        size_t code_size;
        uint32_t* code = util_compile_shader_code(GLSLANG_STAGE_COMPUTE, &code_size, create_info->shader_source, "compute_shader");
        
        if(code == NULL) {
            LOG_ERROR("Failed to compile shader");
            return NULL;
        }

        plan->modules.push_back(ctx->devices[i].createShaderModule(
            vk::ShaderModuleCreateInfo()
            .setCodeSize(code_size)
            .setPCode(code)
        ));

        free(code);

        std::vector<vk::DescriptorSetLayoutBinding> bindings;

        for (int j = 0; j < create_info->binding_count; j++) {
            if(create_info->descriptorTypes[j] != DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                LOG_ERROR("Only storage buffers are supported for now");
                return NULL;
            }

            bindings.push_back(
                vk::DescriptorSetLayoutBinding()
                .setBinding(j)
                .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                .setDescriptorCount(1)
                .setStageFlags(vk::ShaderStageFlagBits::eCompute)
            );

            plan->poolSizes[i].push_back(
                vk::DescriptorPoolSize()
                .setType(vk::DescriptorType::eStorageBuffer)
                .setDescriptorCount(1)
            );
        }

        plan->descriptorSetLayouts.push_back(ctx->devices[i].createDescriptorSetLayout(
            vk::DescriptorSetLayoutCreateInfo()
            .setBindings(bindings)
        ));

        plan->pipelineLayouts.push_back(ctx->devices[i].createPipelineLayout(
            vk::PipelineLayoutCreateInfo()
            .setSetLayouts(plan->descriptorSetLayouts[i])
            .setPushConstantRanges(
                vk::PushConstantRange()
                .setOffset(0)
                .setSize(create_info->pc_size)
                .setStageFlags(vk::ShaderStageFlagBits::eCompute)
            )
        ));

        auto pipelineResult = ctx->devices[i].createComputePipeline(
            vk::PipelineCache(),
            vk::ComputePipelineCreateInfo()
            .setLayout(plan->pipelineLayouts[i])
            .setStage(
                vk::PipelineShaderStageCreateInfo()
                .setStage(vk::ShaderStageFlagBits::eCompute)
                .setModule(plan->modules[i])
                .setPName("main")
            )
        );

        if(pipelineResult.result != vk::Result::eSuccess) {
            LOG_ERROR("Failed to create compute pipeline");
            return NULL;
        }

        plan->pipelines.push_back(pipelineResult.value);
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
        [](vk::CommandBuffer& cmd_buffer, struct Stage* stage, void* instance_data, int device) {
            LOG_INFO("Executing Compute");

            struct ComputeRecordInfo* my_compute_info = (struct ComputeRecordInfo*)stage->user_data;

            cmd_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, my_compute_info->plan->pipelines[device]);
            
            cmd_buffer.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, 
                my_compute_info->plan->pipelineLayouts[device], 
                0, 
                my_compute_info->descriptor_set->sets[device], 
                nullptr
            );
            
            cmd_buffer.pushConstants(
                my_compute_info->plan->pipelineLayouts[device], 
                vk::ShaderStageFlagBits::eCompute, 
                0, 
                my_compute_info->pc_size, 
                instance_data
            );

            cmd_buffer.dispatch(my_compute_info->blocks_x, my_compute_info->blocks_y, my_compute_info->blocks_z);
        },
        my_compute_info,
        plan->pc_size,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    });
}