import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import List

from functools import lru_cache

import numpy as np

cooley_tukey_shuffle_maps = {
    2: [0, 1],
    4: [0, 2, 1, 3],
    8: [0, 4, 2, 6, 1, 5, 3, 7]
}

def stockham_shared_buffer(sdata: vc.Buffer[vc.c64], local_vars, output_offset: int, input_offset: int, index: int, N: int, N_total: int):
    vc.memory_barrier()
    vc.barrier()

    for i in range(4):
        # read odd value
        local_vars[i + 4][:] = sdata[input_offset + index*4 + i + N_total // 2]
        # calculate twiddle factor
        local_vars[i][:] = vc.complex_from_euler_angle(-2 * np.pi * (index*4 + i) / N)
        # do multiplication
        local_vars[i + 4][:] = vc.mult_c64(local_vars[i], local_vars[i + 4])
        
        # read even value
        local_vars[i][:] = sdata[input_offset + index*4 + i]
    
    vc.memory_barrier()
    vc.barrier()

    for i in range(4):
        sdata[output_offset + index*4 + i] = local_vars[i] + local_vars[i + 4]
        sdata[output_offset + index*4 + i + N//2] = local_vars[i] - local_vars[i + 4]

def cooley_tukey_local_vars(
        local_vars: List[vc.Const[vc.c64]],
        offset: int,
        index: int, 
        N: int, 
        N_total: int,
        even_register: vc.Const[vc.c64],
        odd_register: vc.Const[vc.c64],
        factor_register: vc.Const[vc.c64]):
    
    factor_register.x = -2 * np.pi * index / N
    factor_register[:] = vc.complex_from_euler_angle(factor_register.x)

    even_register[:] = local_vars[offset + index]
    odd_register[:] = vc.mult_c64(factor_register, local_vars[offset + index + N//2])

    local_vars[offset + index][:] = even_register + odd_register
    local_vars[offset + index + N//2][:] = even_register - odd_register

def local_var_fft(local_vars, N):
    even_register = vc.new(vc.c64, var_name="even_register")
    odd_register = vc.new(vc.c64, var_name="odd_register")
    factor_register = vc.new(vc.c64, var_name="factor_register")

    total_count = min(N, 8)

    if N > 1:
        cooley_tukey_local_vars(local_vars, 0, 0, 2, total_count, even_register, odd_register, factor_register)
    if N > 2:
        cooley_tukey_local_vars(local_vars, 2, 0, 2, total_count, even_register, odd_register, factor_register)
    if N > 4:
        cooley_tukey_local_vars(local_vars, 4, 0, 2, total_count, even_register, odd_register, factor_register)
        cooley_tukey_local_vars(local_vars, 6, 0, 2, total_count, even_register, odd_register, factor_register)

    if N > 2:
        cooley_tukey_local_vars(local_vars, 0, 0, 4, total_count, even_register, odd_register, factor_register)
        cooley_tukey_local_vars(local_vars, 0, 1, 4, total_count, even_register, odd_register, factor_register)
    if N > 4:
        cooley_tukey_local_vars(local_vars, 4, 0, 4, total_count, even_register, odd_register, factor_register)
        cooley_tukey_local_vars(local_vars, 4, 1, 4, total_count, even_register, odd_register, factor_register)

    if N > 4:
        cooley_tukey_local_vars(local_vars, 0, 0, 8, total_count, even_register, odd_register, factor_register)
        cooley_tukey_local_vars(local_vars, 0, 1, 8, total_count, even_register, odd_register, factor_register)
        cooley_tukey_local_vars(local_vars, 0, 2, 8, total_count, even_register, odd_register, factor_register)
        cooley_tukey_local_vars(local_vars, 0, 3, 8, total_count, even_register, odd_register, factor_register)

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

    tid = vc.local_invocation().x.copy("tid")

    batch_number = vc.workgroup().y
    batch_offset = (batch_number * batch_input_stride).copy()

    local_var_count = min(N, 8)
    shuffle_map = cooley_tukey_shuffle_maps[local_var_count]
    local_vars = [None] * local_var_count

    for i in range(local_var_count):
        local_vars[shuffle_map[i]] = vc.new(vc.c64, buffer[batch_offset + i * (N // local_var_count) + tid], var_name=f"local_{i}")

    local_var_fft(local_vars, N)
    
    max_radix_power = int(np.round(np.log2(N)))
    sdata = vc.shared_buffer(vc.c64, N, "sdata")

    for i in range(local_var_count):
        sdata[tid * local_var_count + i] = local_vars[i]

    for radix_power in range(4, max_radix_power + 1):
        series_length = 2 ** radix_power

        block_index = tid % (series_length // 8)
        block_id = tid / (series_length // 8)

        stockham_shared_buffer(
            sdata,
            local_vars,
            block_id * series_length, 
            block_id * (series_length // 2), 
            block_index, 
            series_length,
            N
        )
    
    vc.memory_barrier()
    vc.barrier()

    for i in range(local_var_count):
        buffer[batch_offset + tid * local_var_count + i] = sdata[tid * local_var_count + i]

    vc.set_global_builder(old_builder)

    return vd.ShaderObject(name, builder.build(name), signature, local_size=(max(1, N // 8), 1, 1))