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
    int omit_depth);
void stage_fft_record_extern(struct CommandList* command_list, struct FFTPlan* plan, struct Buffer* buffer, int inverse);

#endif // _STAGE_FFT_H_