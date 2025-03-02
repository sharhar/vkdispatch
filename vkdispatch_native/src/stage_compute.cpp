#include "../include/internal.hh"

#include <chrono>
#include <thread>

#include <glslang_c_interface.h>

static uint32_t* glsl_to_spirv_util(glslang_stage_t stage, glslang_resource_t* resource_limits, size_t* size, const char* shader_source, const char* shader_name) {
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
	input.resource = resource_limits;

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
    plan->poolSizes = new VkDescriptorPoolSize[plan->binding_count];
    plan->bindings = new VkDescriptorSetLayoutBinding[plan->binding_count];
    
    uint64_t descriptor_set_layouts_handle = ctx->handle_manager->register_device_handle("DescriptorSetLayouts");
    uint64_t pipeline_layouts_handle = ctx->handle_manager->register_device_handle("PipelineLayouts");
    uint64_t pipelines_handle = ctx->handle_manager->register_device_handle("Pipelines");

    plan->descriptorSetLayouts_handle = descriptor_set_layouts_handle;
    plan->pipelineLayouts_handle = pipeline_layouts_handle;
    plan->pipelines_handle = pipelines_handle;

    for (int j = 0; j < plan->binding_count; j++) {
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


        plan->bindings[j] = {};
        plan->bindings[j].binding = j;
        plan->bindings[j].descriptorType = descriptorType;
        plan->bindings[j].descriptorCount = 1;
        plan->bindings[j].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        plan->bindings[j].pImmutableSamplers = NULL;
        
        plan->poolSizes[j] = {};
        plan->poolSizes[j].type = descriptorType;
        plan->poolSizes[j].descriptorCount = 1;
    }

    ctx->glslang_mutex.lock();

    plan->code = glsl_to_spirv_util(
        GLSLANG_STAGE_COMPUTE, 
        reinterpret_cast<glslang_resource_t*>(ctx->glslang_resource_limits), 
        &plan->code_size, 
        create_info->shader_source, 
        create_info->shader_name
    );

    ctx->glslang_mutex.unlock();
    
    if(plan->code == NULL) {
        set_error("Failed to compile compute shader!");
        return NULL;
    }

    uint32_t* code = plan->code;
    size_t code_size = plan->code_size;
    unsigned int pc_size = create_info->pc_size;
    VkDescriptorSetLayoutBinding* bindings = plan->bindings;
    unsigned int binding_count = plan->binding_count;

    command_list_record_command(ctx->command_list, 
        "compute-init",
        0,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        [ctx, code, code_size, pc_size, descriptor_set_layouts_handle, pipeline_layouts_handle, pipelines_handle, bindings, binding_count]
        (VkCommandBuffer cmd_buffer, int device_index, int stream_index, int recorder_index, void* pc_data) {
            ctx->handle_manager->set_handle_per_device(device_index, descriptor_set_layouts_handle, 
            [ctx, bindings, binding_count, stream_index, recorder_index](int device_index) {
                LOG_VERBOSE("Creating Descriptor Set Layout for device %d on stream %d recorder %d", device_index, stream_index, recorder_index);

                VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
                descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                descriptorSetLayoutCreateInfo.pNext = nullptr;
                descriptorSetLayoutCreateInfo.flags = 0;
                descriptorSetLayoutCreateInfo.bindingCount = binding_count;
                descriptorSetLayoutCreateInfo.pBindings = bindings;

                VkDescriptorSetLayout descriptorSetLayout;
                VK_CALL_RETURN(vkCreateDescriptorSetLayout(ctx->devices[device_index], &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayout), (uint64_t)0);

                return (uint64_t)descriptorSetLayout;
            });

            ctx->handle_manager->set_handle_per_device(device_index, pipeline_layouts_handle,
            [ctx, descriptor_set_layouts_handle, pc_size, stream_index, recorder_index](int device_index) {
                LOG_VERBOSE("Creating Pipeline Layout for device %d on stream %d recorder %d", device_index, stream_index, recorder_index);

                VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
                pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
                pipelineLayoutCreateInfo.pNext = nullptr;
                pipelineLayoutCreateInfo.flags = 0;
                pipelineLayoutCreateInfo.setLayoutCount = 1;

                LOG_VERBOSE("Descriptor Set Layout Handle: %d", descriptor_set_layouts_handle);

                pipelineLayoutCreateInfo.pSetLayouts = (VkDescriptorSetLayout*)ctx->handle_manager->get_handle_pointer_no_lock(stream_index, descriptor_set_layouts_handle);
                pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
                pipelineLayoutCreateInfo.pPushConstantRanges = nullptr;

                LOG_VERBOSE("Push Constant Size: %d", pc_size);

                VkPushConstantRange pushConstantRange = {};
                pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                pushConstantRange.offset = 0;
                pushConstantRange.size = pc_size;
                if(pc_size > 0) {
                    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
                    pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
                }

                LOG_VERBOSE("Creating Pipeline Layout for device %d on stream %d", device_index, stream_index);

                VkPipelineLayout pipelineLayout;
                VK_CALL_RETURN(vkCreatePipelineLayout(ctx->devices[device_index], &pipelineLayoutCreateInfo, NULL, &pipelineLayout), (uint64_t)0);

                LOG_VERBOSE("Pipeline Layout Handle: %d", (uint64_t)pipelineLayout);

                return (uint64_t)pipelineLayout;
            });

            ctx->handle_manager->set_handle_per_device(device_index, pipelines_handle,
            [ctx, code, code_size, pipeline_layouts_handle, stream_index](int device_index) {
                LOG_VERBOSE("Creating Pipeline for device %d on stream %d", device_index, stream_index);

                VkComputePipelineCreateInfo pipelineCreateInfo = {};
                pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                pipelineCreateInfo.pNext = nullptr;
                pipelineCreateInfo.flags = 0;
                pipelineCreateInfo.layout = (VkPipelineLayout)ctx->handle_manager->get_handle_no_lock(stream_index, pipeline_layouts_handle);
                pipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
                pipelineCreateInfo.basePipelineIndex = -1;

                VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
                shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
                shaderModuleCreateInfo.pNext = nullptr;
                shaderModuleCreateInfo.flags = 0;
                shaderModuleCreateInfo.codeSize = code_size;
                shaderModuleCreateInfo.pCode = code;
                
                VkShaderModule shaderModule;
                VK_CALL_RETURN(vkCreateShaderModule(ctx->devices[device_index], &shaderModuleCreateInfo, NULL, &shaderModule), (uint64_t)0);       

                pipelineCreateInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                pipelineCreateInfo.stage.pNext = nullptr;
                pipelineCreateInfo.stage.flags = 0;
                pipelineCreateInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                pipelineCreateInfo.stage.module = shaderModule;
                pipelineCreateInfo.stage.pName = "main";
                pipelineCreateInfo.stage.pSpecializationInfo = nullptr;

                VkPipeline pipeline;
                VK_CALL_RETURN(vkCreateComputePipelines(ctx->devices[device_index], VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &pipeline), (uint64_t)0);
            
                vkDestroyShaderModule(ctx->devices[device_index], shaderModule, NULL);

                return (uint64_t)pipeline;
            });
        }
    );

    int submit_index = -2;
    command_list_submit_extern(ctx->command_list, NULL, 1, &submit_index, 1, NULL, RECORD_TYPE_SYNC);
    command_list_reset_extern(ctx->command_list);
    RETURN_ON_ERROR(NULL)

    /*

    for (int i = 0; i < ctx->deviceCount; i++) {
        LOG_INFO("Creating Compute Plan for device %d", i);

        VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
        shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderModuleCreateInfo.pNext = nullptr;
        shaderModuleCreateInfo.flags = 0;
        shaderModuleCreateInfo.codeSize = plan->code_size;
        shaderModuleCreateInfo.pCode = plan->code;
        VK_CALL_RETNULL(vkCreateShaderModule(ctx->devices[i], &shaderModuleCreateInfo, NULL, &plan->modules[i]));        

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

            VkDescriptorSetLayoutBinding binding = {};
            binding.binding = j;
            binding.descriptorType = descriptorType;
            binding.descriptorCount = 1;
            binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            binding.pImmutableSamplers = NULL;
            bindings.push_back(binding);

            VkDescriptorPoolSize poolSize = {};
            poolSize.type = descriptorType;
            poolSize.descriptorCount = 1;
            plan->poolSizes[i].push_back(poolSize);
        }

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
        descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCreateInfo.pNext = nullptr;
        descriptorSetLayoutCreateInfo.flags = 0;
        descriptorSetLayoutCreateInfo.bindingCount = bindings.size();
        descriptorSetLayoutCreateInfo.pBindings = bindings.data();
        VK_CALL_RETNULL(vkCreateDescriptorSetLayout(ctx->devices[i], &descriptorSetLayoutCreateInfo, NULL, &plan->descriptorSetLayouts[i]));

        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
        pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutCreateInfo.pNext = nullptr;
        pipelineLayoutCreateInfo.flags = 0;
        pipelineLayoutCreateInfo.setLayoutCount = 1;
        pipelineLayoutCreateInfo.pSetLayouts = &plan->descriptorSetLayouts[i];
        pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
        pipelineLayoutCreateInfo.pPushConstantRanges = nullptr;    

        VkPushConstantRange pushConstantRange = {};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = create_info->pc_size;
        if(create_info->pc_size > 0) {
            pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
            pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
        }
        VK_CALL_RETNULL(vkCreatePipelineLayout(ctx->devices[i], &pipelineLayoutCreateInfo, NULL, &plan->pipelineLayouts[i]));

        VkComputePipelineCreateInfo pipelineCreateInfo = {};
        pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineCreateInfo.pNext = nullptr;
        pipelineCreateInfo.flags = 0;
        pipelineCreateInfo.layout = plan->pipelineLayouts[i];
        pipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineCreateInfo.basePipelineIndex = -1;

        pipelineCreateInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineCreateInfo.stage.pNext = nullptr;
        pipelineCreateInfo.stage.flags = 0;
        pipelineCreateInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineCreateInfo.stage.module = plan->modules[i];
        pipelineCreateInfo.stage.pName = "main";
        pipelineCreateInfo.stage.pSpecializationInfo = nullptr;

        VK_CALL_RETNULL(vkCreateComputePipelines(ctx->devices[i], VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &plan->pipelines[i]));
    }

    */

    return plan;
}

void stage_compute_record_extern(struct CommandList* command_list, struct ComputePlan* plan, struct DescriptorSet* descriptor_set, unsigned int blocks_x, unsigned int blocks_y, unsigned int blocks_z) {
    uint64_t sets_handle = 0;
    if(descriptor_set != NULL)
        sets_handle = descriptor_set->sets_handle;

    uint64_t pipelineLayouts_handle = plan->pipelineLayouts_handle;
    uint64_t pipelines_handle = plan->pipelines_handle;

    struct Context* ctx = plan->ctx;

    size_t pc_size = plan->pc_size;

    command_list_record_command(command_list,
        "compute-stage",
        pc_size,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        [ctx, pipelineLayouts_handle, pipelines_handle, sets_handle, pc_size, blocks_x, blocks_y, blocks_z](VkCommandBuffer cmd_buffer, int device_index, int stream_index, int recorder_index, void* pc_data) {
            vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->handle_manager->get_handle(stream_index, pipelines_handle));

            VkPipelineLayout pipelineLayout = (VkPipelineLayout)ctx->handle_manager->get_handle(stream_index, pipelineLayouts_handle);

            if(sets_handle != 0) {
                VkDescriptorSet desc_set = (VkDescriptorSet)ctx->handle_manager->get_handle(stream_index, sets_handle);

                vkCmdBindDescriptorSets(
                    cmd_buffer,
                    VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipelineLayout,
                    0,
                    1,
                    &desc_set,
                    0,
                    NULL
                );
            }

            if(pc_size > 0)
                vkCmdPushConstants(
                    cmd_buffer, 
                    pipelineLayout,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    0,
                    pc_size,
                    pc_data
                );

            vkCmdDispatch(cmd_buffer, blocks_x, blocks_y, blocks_z);
        }
    );
}