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

struct FFTPlan* stage_fft_plan_create_extern(
    struct Context* ctx, 
    unsigned long long dims, 
    unsigned long long rows, 
    unsigned long long cols, 
    unsigned long long depth, 
    unsigned long long buffer_size, 
    unsigned int do_r2c,
    int omit_rows,
    int omit_cols,
    int omit_depth,
    int normalize,
    unsigned long long pad_left_rows, unsigned long long pad_right_rows,
    unsigned long long pad_left_cols, unsigned long long pad_right_cols,
    unsigned long long pad_left_depth, unsigned long long pad_right_depth,
    int frequency_zeropadding,
    int kernel_num,
    int kernel_convolution,
    unsigned long long input_buffer_size);

void stage_fft_record_extern(
    struct CommandList* command_list, 
    struct FFTPlan* plan, 
    struct Buffer* buffer, 
    int inverse, 
    struct Buffer* kernel,
    struct Buffer* input_buffer);

#endif // _STAGE_FFT_H_