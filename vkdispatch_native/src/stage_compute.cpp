#include "../include/internal.hh"

#include <glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>

static uint32_t* glsl_to_spirv_util(glslang_stage_t stage, size_t* size, const char* shader_source, const char* shader_name) {
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
        set_error(input.code);
        //LOG_ERROR("%s", input.code);
        glslang_shader_delete(shader);
        return NULL;
    }

    if (!glslang_shader_parse(shader, &input)) {
        LOG_ERROR("GLSL parsing failed %s", shader_name);
        LOG_ERROR("%s", glslang_shader_get_info_log(shader));
        LOG_ERROR("%s", glslang_shader_get_info_debug_log(shader));
        set_error(glslang_shader_get_preprocessed_code(shader));
        //LOG_ERROR("%s", glslang_shader_get_preprocessed_code(shader));
        glslang_shader_delete(shader);
        return NULL;
    }

    glslang_program_t* program = glslang_program_create();
    glslang_program_add_shader(program, shader);

    if (!glslang_program_link(program, GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT)) {
        LOG_ERROR("GLSL linking failed %s", shader_name);
        LOG_ERROR("%s", glslang_program_get_info_log(program));
        LOG_ERROR("%s", glslang_program_get_info_debug_log(program));
        set_error(glslang_shader_get_preprocessed_code(shader));
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
        LOG_ERROR("(%s) %s\n", shader_name, spirv_messages);

    glslang_program_delete(program);
    glslang_shader_delete(shader);

    return words;
}

struct ComputePlan* stage_compute_plan_create_extern(struct Context* ctx, struct ComputePlanCreateInfo* create_info) {
    struct ComputePlan* plan = new struct ComputePlan();
    LOG_INFO("Creating Compute Plan with handle %p", plan);
    
    plan->ctx = ctx;
    plan->pc_size = create_info->pc_size;
    plan->binding_count = create_info->binding_count;
    plan->poolSizes.resize(ctx->deviceCount);
    plan->modules.resize(ctx->deviceCount);
    plan->descriptorSetLayouts.resize(ctx->deviceCount);
    plan->pipelineLayouts.resize(ctx->deviceCount);
    plan->pipelines.resize(ctx->deviceCount);

    for (int i = 0; i < ctx->deviceCount; i++) {
        LOG_INFO("Creating Compute Plan for device %d", i);

        size_t code_size;
        uint32_t* code = glsl_to_spirv_util(GLSLANG_STAGE_COMPUTE, &code_size, create_info->shader_source, create_info->shader_name);
        
        if(code == NULL) {
            //set_error("Failed to compile compute shader!");
            return NULL;
        }

        VkShaderModuleCreateInfo shaderModuleCreateInfo;
        memset(&shaderModuleCreateInfo, 0, sizeof(VkShaderModuleCreateInfo));
        shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderModuleCreateInfo.codeSize = code_size;
        shaderModuleCreateInfo.pCode = code;
        VK_CALL_RETNULL(vkCreateShaderModule(ctx->devices[i], &shaderModuleCreateInfo, NULL, &plan->modules[i]));

        free(code);

        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (int j = 0; j < create_info->binding_count; j++) {
            if(create_info->descriptorTypes[j] != DESCRIPTOR_TYPE_STORAGE_BUFFER &&
               create_info->descriptorTypes[j] != DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
               create_info->descriptorTypes[j] != DESCRIPTOR_TYPE_SAMPLER) {
                LOG_ERROR("Only storage and uniform buffers are supported for now");
                return NULL;
            }

            VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            if(create_info->descriptorTypes[j] == DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            else if(create_info->descriptorTypes[j] == DESCRIPTOR_TYPE_SAMPLER)
                descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

            VkDescriptorSetLayoutBinding binding;
            memset(&binding, 0, sizeof(VkDescriptorSetLayoutBinding));
            binding.binding = j;
            binding.descriptorType = descriptorType;
            binding.descriptorCount = 1;
            binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings.push_back(binding);

            VkDescriptorPoolSize poolSize;
            memset(&poolSize, 0, sizeof(VkDescriptorPoolSize));
            poolSize.type = descriptorType;
            poolSize.descriptorCount = 1;
            plan->poolSizes[i].push_back(poolSize);
        }

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo;
        memset(&descriptorSetLayoutCreateInfo, 0, sizeof(VkDescriptorSetLayoutCreateInfo));
        descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCreateInfo.bindingCount = bindings.size();
        descriptorSetLayoutCreateInfo.pBindings = bindings.data();
        VK_CALL_RETNULL(vkCreateDescriptorSetLayout(ctx->devices[i], &descriptorSetLayoutCreateInfo, NULL, &plan->descriptorSetLayouts[i]));

        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo;
        memset(&pipelineLayoutCreateInfo, 0, sizeof(VkPipelineLayoutCreateInfo));
        pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutCreateInfo.setLayoutCount = 1;
        pipelineLayoutCreateInfo.pSetLayouts = &plan->descriptorSetLayouts[i];

        VkPushConstantRange pushConstantRange;
        memset(&pushConstantRange, 0, sizeof(VkPushConstantRange));
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = create_info->pc_size;
        if(create_info->pc_size > 0) {
            pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
            pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
        }
        VK_CALL_RETNULL(vkCreatePipelineLayout(ctx->devices[i], &pipelineLayoutCreateInfo, NULL, &plan->pipelineLayouts[i]));

        VkComputePipelineCreateInfo pipelineCreateInfo;
        memset(&pipelineCreateInfo, 0, sizeof(VkComputePipelineCreateInfo));
        pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineCreateInfo.layout = plan->pipelineLayouts[i];
        pipelineCreateInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineCreateInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineCreateInfo.stage.module = plan->modules[i];
        pipelineCreateInfo.stage.pName = "main";
        VK_CALL_RETNULL(vkCreateComputePipelines(ctx->devices[i], VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &plan->pipelines[i]));
    }

    return plan;
}


void stage_compute_record_extern(struct CommandList* command_list, struct ComputePlan* plan, struct DescriptorSet* descriptor_set, unsigned int blocks_x, unsigned int blocks_y, unsigned int blocks_z) {
    struct CommandInfo command = {};
    command.type = COMMAND_TYPE_COMPUTE;
    command.pipeline_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    command.info.compute_info.plan = plan;
    command.info.compute_info.descriptor_set = descriptor_set;
    command.info.compute_info.blocks_x = blocks_x;
    command.info.compute_info.blocks_y = blocks_y;
    command.info.compute_info.blocks_z = blocks_z;
    command.info.compute_info.pc_size = plan->pc_size;

    command_list_record_command(command_list, command);
}

void stage_compute_plan_exec_internal(VkCommandBuffer cmd_buffer, const struct ComputeRecordInfo& info, void* instance_data, int device_index, int stream_index) {
    vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, info.plan->pipelines[device_index]);

    if(info.descriptor_set != NULL)
        vkCmdBindDescriptorSets(
            cmd_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            info.plan->pipelineLayouts[device_index],
            0,
            1,
            &info.descriptor_set->sets[stream_index],
            0,
            NULL
        );

    if(info.pc_size > 0)
        vkCmdPushConstants(
            cmd_buffer, 
            info.plan->pipelineLayouts[device_index],
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            info.pc_size,
            instance_data
        );

    vkCmdDispatch(cmd_buffer, info.blocks_x, info.blocks_y, info.blocks_z);
}