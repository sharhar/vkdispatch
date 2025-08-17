#include "barrier_manager.hh"
#include <cstring>

#include "../objects/buffer.hh"

BarrierManager::BarrierManager() {

}

void BarrierManager::record_barriers(VkCommandBuffer cmd_buffer, struct BufferBarrierInfo* buffer_barrier_infos, int buffer_barrier_count, int queue_index) {
    int barrier_count = 0;

    VkBufferMemoryBarrier* buffer_barriers = (VkBufferMemoryBarrier*)malloc(sizeof(VkBufferMemoryBarrier) * buffer_barrier_count);
    memset(buffer_barriers, 0, sizeof(VkBufferMemoryBarrier) * buffer_barrier_count);

    for(int i = 0; i < buffer_barrier_count; i++) {
        struct Buffer* buffer_id = buffer_barrier_infos[i].buffer_id;

        if(!buffer_barrier_infos[i].read && !buffer_barrier_infos[i].write) {
            continue; // No need to add a barrier if the buffer is not being read or written to
        }

        // Don't add a barrier if the buffer is not in the map
        if(buffer_states.find(buffer_id) == buffer_states.end()) {
            buffer_states[buffer_id] = std::make_pair(buffer_barrier_infos[i].read, buffer_barrier_infos[i].write);
            continue;
        }

        if(!buffer_states[buffer_id].second && !buffer_barrier_infos[i].write) {
            continue; // No need to add a barrier if the buffer is not being written to
        }

        VkAccessFlags srcAccessMask = 0;
        srcAccessMask |= buffer_states[buffer_id].first ? VK_ACCESS_SHADER_READ_BIT : 0;
        srcAccessMask |= buffer_states[buffer_id].second ? VK_ACCESS_SHADER_WRITE_BIT : 0;

        VkAccessFlags dstAccessMask = 0;
        dstAccessMask |= buffer_barrier_infos[i].read ? VK_ACCESS_SHADER_READ_BIT : 0;
        dstAccessMask |= buffer_barrier_infos[i].write ? VK_ACCESS_SHADER_WRITE_BIT : 0;

        buffer_barriers[barrier_count].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        buffer_barriers[barrier_count].pNext = nullptr;
        buffer_barriers[barrier_count].srcAccessMask = srcAccessMask;
        buffer_barriers[barrier_count].dstAccessMask = dstAccessMask;
        buffer_barriers[barrier_count].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        buffer_barriers[barrier_count].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        buffer_barriers[barrier_count].buffer = buffer_id->buffers[queue_index];
        buffer_barriers[barrier_count].offset = 0;
        buffer_barriers[barrier_count].size = VK_WHOLE_SIZE;

        buffer_states[buffer_id] = std::make_pair(buffer_barrier_infos[i].read, buffer_barrier_infos[i].write);

        barrier_count++;
    }

    if (barrier_count != 0) {
        vkCmdPipelineBarrier(
            cmd_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr,
            barrier_count, buffer_barriers,
            0, nullptr
        );
    }

    free(buffer_barriers);
}

void BarrierManager::reset() {
    buffer_states.clear();
}
