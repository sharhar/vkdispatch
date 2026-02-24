import vkdispatch.codegen as vc
from .resources import FFTResources

from typing import List, Union

from .._compat import numpy_compat as npc

def get_angle_factor(inverse: bool) -> float:
    return 2 * npc.pi * (1 if inverse else -1)

def _apply_right_angle_twiddle(resources: FFTResources, register: vc.ShaderVariable, angle_int: int) -> bool:
    if angle_int == 0:
        return True

    if angle_int == 1:
        resources.radix_registers[0].real = register.real
        register.real = -register.imag
        register.imag = resources.radix_registers[0].real
        return True

    if angle_int == -1:
        resources.radix_registers[0].real = register.real
        register.real = register.imag
        register.imag = -resources.radix_registers[0].real
        return True

    if angle_int == 2 or angle_int == -2:
        register[:] = -register
        return True

    return False

def _apply_constant_twiddle(resources: FFTResources, register: vc.ShaderVariable, omega: complex) -> bool:
    scaled_angle = 2 * npc.angle(omega) / npc.pi
    rounded_angle = npc.round(scaled_angle)

    if abs(scaled_angle - rounded_angle) >= 1e-8:
        return False

    return _apply_right_angle_twiddle(resources, register, int(rounded_angle))

def _apply_twiddle_to_register(
        resources: FFTResources,
        register: vc.ShaderVariable,
        twiddle: Union[complex, vc.ShaderVariable]):
    if isinstance(twiddle, complex):
        if _apply_constant_twiddle(resources, register, twiddle):
            return

        twiddle = vc.to_dtype(register.var_type, twiddle.real, twiddle.imag)

    resources.radix_registers[0][:] = vc.mult_complex(register, twiddle)
    register[:] = resources.radix_registers[0]

def radix_P(resources: FFTResources, inverse: bool, register_list: List[vc.ShaderVariable]):
    assert len(register_list) <= len(resources.radix_registers), "Too many registers for radix_P"

    if len(register_list) == 1:
        return
    
    if len(register_list) == 2:
        vc.comment("Radix-2 butterfly base case", preceding_new_line=False)
        resources.radix_registers[0][:] = register_list[1]
        register_list[1][:] = register_list[0] - resources.radix_registers[0]
        register_list[0][:] = register_list[0] + resources.radix_registers[0]
        return

    vc.comment(f"Radix-{len(register_list)} DFT", preceding_new_line=False)

    angle_factor = get_angle_factor(inverse)

    for i in range(0, len(register_list)):
        for j in range(0, len(register_list)):
            if j == 0:
                resources.radix_registers[i][:] = register_list[j]
                continue

            if i == 0:
                resources.radix_registers[i] += register_list[j]
                continue

            if i * j == len(register_list) // 2 and len(register_list) % 2 == 0:
                resources.radix_registers[i] -= register_list[j]
                continue

            omega = npc.exp_complex(1j * angle_factor * i * j / len(register_list))
            typed_omega = vc.to_dtype(register_list[j].var_type, omega.real, omega.imag)
            resources.omega_register[:] = vc.mult_complex(register_list[j], typed_omega)
            resources.radix_registers[i] += resources.omega_register

    for i in range(0, len(register_list)):
        register_list[i][:] = resources.radix_registers[i]

def apply_twiddle_factors(
        resources: FFTResources,
        inverse: bool,
        register_list: List[vc.ShaderVariable],
        twiddle_index: Union[int, vc.ShaderVariable] = 0,
        twiddle_N: int = 1):

    if isinstance(twiddle_index, int) and twiddle_index == 0:
        return

    twiddle_index_str = str(twiddle_index) if isinstance(twiddle_index, int) else twiddle_index.resolve()
    vc.comment(f"""Applying Cooley-Tukey inter-stage twiddle factors before the next butterfly pass.
Twiddle domain size: N = {twiddle_N}. Twiddle index source: {twiddle_index_str}.
For each non-DC lane i>0, multiply by W_N^(i * twiddle_index).
This phase-aligns each sub-FFT with its parent decomposition stage.""")

    angle_factor = get_angle_factor(inverse)

    for i in range(len(register_list)):
        if i == 0:
            continue
        
        if isinstance(twiddle_index, int):
            if twiddle_index == 0:
                continue

            omega = npc.exp_complex(1j * angle_factor * i * twiddle_index / twiddle_N)

            _apply_twiddle_to_register(resources, register_list[i], omega)
            continue

        angle_scale = vc.to_dtype(resources.omega_register.real.var_type, angle_factor * i / twiddle_N)
        twiddle_scale = vc.to_dtype(resources.omega_register.real.var_type, twiddle_index)
        resources.omega_register.real = angle_scale * twiddle_scale
        resources.omega_register[:] = vc.complex_from_euler_angle(resources.omega_register.real)
        resources.radix_registers[0][:] = vc.mult_complex(register_list[i], resources.omega_register)
        register_list[i][:] = resources.radix_registers[0]

def radix_composite(
        resources: FFTResources,
        inverse: bool,
        register_list: List[vc.ShaderVariable],
        primes: List[int],
        twiddle_index: Union[int, vc.ShaderVariable] = 0,
        twiddle_N: int = 1):
    if len(register_list) == 1:
        return
    
    N = len(register_list)

    assert N == npc.prod(primes), "Product of primes must be equal to the number of registers"

    vc.comment(f"""Starting mixed-radix FFT decomposition for this invocation on {N} register samples.
Radix factorization sequence: {primes}.
At each level: partition lanes into stage-local sub-sequences, apply twiddles,
run radix-P butterflies, then reassemble in stride-consistent order for downstream stages.""")

    apply_twiddle_factors(
        resources=resources,
        inverse=inverse,
        register_list=register_list,
        twiddle_index=twiddle_index,
        twiddle_N=twiddle_N
    )

    output_stride = 1

    for prime in primes:
        sub_squences = [register_list[i::N//prime] for i in range(N//prime)]

        block_width = output_stride * prime

        for i in range(0, N // prime):
            inner_block_offset = i % output_stride
            block_index = (i * prime) // block_width

            apply_twiddle_factors(resources, inverse, sub_squences[i], twiddle_index=inner_block_offset, twiddle_N=block_width)
            radix_P(resources, inverse, sub_squences[i])
            
            sub_sequence_offset = block_index * block_width + inner_block_offset

            for j in range(prime):
                register_list[sub_sequence_offset + j * output_stride] = sub_squences[i][j]
        
        output_stride *= prime   

    return register_list
