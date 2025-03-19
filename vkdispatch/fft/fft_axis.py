import vkdispatch as vd
import vkdispatch.codegen as vc

from functools import lru_cache

import numpy as np

def stockham_sdata(sdata: vc.Buffer[vc.c64], output_offset: int, input_offset: int, index: int, N: int, N_total: int):
    factor = vc.complex_from_euler_angle(-2 * np.pi * index / N)

    vc.memory_barrier()
    vc.barrier()

    even_value = sdata[input_offset + index].copy()
    odd_value = sdata[input_offset + index + N_total//2].copy()

    odd_factor = vc.mult_c64(factor, odd_value).copy()
    
    vc.memory_barrier()
    vc.barrier()

    sdata[output_offset + index] = even_value + odd_factor
    sdata[output_offset + index + N//2] = even_value - odd_factor

@lru_cache(maxsize=None)
def make_fft_stage(
        N: int, 
        stride: int = 1,
        batch_input_stride: int = 1,
        batch_output_stride: int = 1,
        name: str = None):
    
    assert N & (N-1) == 0, "Input length must be a power of 2"
    assert stride == 1, "Only stride 1 is supported for now"

    if name is None:
        name = f"fft_stage_{N}_{stride}"

    builder = vc.ShaderBuilder(enable_exec_bounds=False)
    old_builder = vc.set_global_builder(builder)

    signature = vd.ShaderSignature.from_type_annotations(builder, [vc.Buffer[vc.c64]])
    input_variables = signature.get_variables()

    buffer = input_variables[0]

    sdata = vc.shared_buffer(vc.c64, N, "sdata")

    tid = vc.local_invocation().x.copy("tid")

    batch_number = vc.workgroup().y

    sdata[tid] = buffer[batch_number * batch_input_stride + tid]
    sdata[tid + N//2] = buffer[batch_number * batch_input_stride + tid + N//2]

    max_radix_power = int(np.round(np.log2(N)))

    for radix_power in range(1, max_radix_power + 1):
        series_length = 2 ** radix_power

        block_index = tid % (series_length // 2)
        block_id = tid / (series_length // 2)

        stockham_sdata(
            sdata,
            block_id * series_length, 
            block_id * (series_length // 2), 
            block_index, 
            series_length,
            N
        )

    buffer[batch_number * batch_output_stride + tid] = sdata[tid]
    buffer[batch_number * batch_output_stride + tid + N//2] = sdata[tid + N//2]

    vc.set_global_builder(old_builder)

    return vd.ShaderObject(name, builder.build(name), signature, local_size=(N // 2, 1, 1))