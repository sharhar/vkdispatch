#ifndef _STAGE_FFT_H_
#define _STAGE_FFT_H_

#include "base.hh"

struct FFTInitRecordInfo {
    struct FFTPlan* plan;
};

struct FFTExecRecordInfo {
    struct FFTPlan* plan;
    struct Buffer* buffer;
    int inverse;
};

struct FFTPlan* stage_fft_plan_create_extern(struct Context* ctx, unsigned long long dims, unsigned long long rows, unsigned long long cols, unsigned long long depth, unsigned long long buffer_size, unsigned int do_r2c);
void stage_fft_record_extern(struct CommandList* command_list, struct FFTPlan* plan, struct Buffer* buffer, int inverse);

void stage_fft_plan_init_internal(const struct FFTInitRecordInfo& info, int device_index, int stream_index, int recorder_index);
void stage_fft_plan_exec_internal(VkCommandBuffer cmd_buffer, const struct FFTExecRecordInfo& info, int device_index, int stream_index, int recorder_index);

#endif // _STAGE_FFT_H_