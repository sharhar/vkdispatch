#ifndef _STAGE_FFT_H_
#define _STAGE_FFT_H_

#include "base.h"

struct FFTRecordInfo {
    struct FFTPlan* plan;
    struct Buffer* buffer;
    int inverse;
};

struct FFTPlan* stage_fft_plan_create_extern(struct Context* ctx, unsigned long long dims, unsigned long long rows, unsigned long long cols, unsigned long long depth, unsigned long long buffer_size);
void stage_fft_record_extern(struct CommandList* command_list, struct FFTPlan* plan, struct Buffer* buffer, int inverse);

void stage_fft_plan_exec_internal(VkCommandBuffer cmd_buffer, const struct FFTRecordInfo& info, int device_index, int stream_index);

#endif // _STAGE_FFT_H_