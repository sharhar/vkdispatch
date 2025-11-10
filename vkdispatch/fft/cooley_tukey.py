import vkdispatch.codegen as vc
from .resources import FFTResources

from typing import List, Union

import numpy as np

def get_angle_factor(inverse: bool) -> float:
    return 2 * np.pi * (1 if inverse else -1)

def radix_P(resources: FFTResources, inverse: bool, register_list: List[vc.ShaderVariable]):
    assert len(register_list) <= len(resources.radix_registers), "Too many registers for radix_P"

    if len(register_list) == 1:
        return
    
    if len(register_list) == 2:
        vc.comment(f"Performing a DFT for Radix-2 FFT")
        resources.radix_registers[0][:] = register_list[1]
        register_list[1][:] = register_list[0] - resources.radix_registers[0]
        register_list[0][:] = register_list[0] + resources.radix_registers[0]
        return

    vc.comment(f"Performing a DFT for Radix-{len(register_list)} FFT")

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

            omega = np.exp(1j * angle_factor * i * j / len(register_list))
            resources.omega_register[:] = vc.mult_complex(register_list[j], omega)
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

    vc.comment(f"Applying Cooley-Tukey twiddle factors for twiddle index {twiddle_index} and twiddle N {twiddle_N}")

    angle_factor = get_angle_factor(inverse)

    if not isinstance(twiddle_index, int):
        resources.omega_register.real = (angle_factor / twiddle_N) * twiddle_index 
        resources.omega_register[:] = vc.complex_from_euler_angle(resources.omega_register.real)
        resources.radix_registers[1][:] = resources.omega_register

    for i in range(len(register_list)):
        if i == 0:
            continue
        
        if isinstance(twiddle_index, int):
            if twiddle_index == 0:
                continue

            omega = np.exp(1j * angle_factor * i * twiddle_index / twiddle_N)

            scaled_angle = 2 * np.angle(omega) / np.pi
            rounded_angle = np.round(scaled_angle)

            if np.abs(scaled_angle - rounded_angle) < 1e-8:
                angle_int = int(rounded_angle)

                if angle_int == 1:
                    resources.omega_register.real = register_list[i].real
                    register_list[i].real = -register_list[i].imag
                    register_list[i].imag = resources.omega_register.real
                elif angle_int == -1:
                    resources.omega_register.real = register_list[i].real
                    register_list[i].real = register_list[i].imag
                    register_list[i].imag = -resources.omega_register.real
                elif angle_int == 2 or angle_int == -2:
                    register_list[i][:] = -register_list[i]
                
                continue

            resources.omega_register[:] = vc.mult_complex(register_list[i], omega)
            register_list[i][:] = resources.omega_register
            continue
        

        resources.radix_registers[0][:] = vc.mult_complex(register_list[i], resources.radix_registers[1])
        register_list[i][:] = resources.radix_registers[0]

        if i < len(register_list) - 1:
            resources.radix_registers[0][:] = vc.mult_complex(resources.omega_register, resources.radix_registers[1])
            resources.radix_registers[1][:] = resources.radix_registers[0]

def radix_composite(resources: FFTResources, inverse: bool, register_list: List[vc.ShaderVariable], primes: List[int]):
    if len(register_list) == 1:
        return
    
    N = len(register_list)

    assert N == np.prod(primes), "Product of primes must be equal to the number of registers"

    vc.comment(f"Performing a Radix-{primes} FFT on {N} registers")

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
