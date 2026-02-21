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
    resources.radix_registers[0][:] = vc.mult_complex(register, twiddle)
    register[:] = resources.radix_registers[0]

def _apply_combined_twiddle_to_register(
        resources: FFTResources,
        register: vc.ShaderVariable,
        base_twiddle: Union[None, complex, vc.ShaderVariable],
        fixed_twiddle: complex):
    if base_twiddle is not None:
        _apply_twiddle_to_register(resources, register, base_twiddle)
    _apply_twiddle_to_register(resources, register, fixed_twiddle)

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

            omega = npc.exp_complex(1j * angle_factor * i * j / len(register_list))
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

    twiddle_index_str = str(twiddle_index) if isinstance(twiddle_index, int) else twiddle_index.resolve()
    vc.comment(f"Applying Cooley-Tukey twiddle factors for twiddle index {twiddle_index_str} and twiddle N {twiddle_N}")

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

        resources.omega_register.real = (angle_factor * i / twiddle_N) * twiddle_index
        resources.omega_register[:] = vc.complex_from_euler_angle(resources.omega_register.real)
        resources.radix_registers[0][:] = vc.mult_complex(register_list[i], resources.omega_register)
        register_list[i][:] = resources.radix_registers[0]

def _radix_composite_fused_power_of_two(
        resources: FFTResources,
        inverse: bool,
        register_list: List[vc.ShaderVariable],
        level_count: int,
        twiddle_index: Union[int, vc.ShaderVariable],
        twiddle_N: int):
    N = len(register_list)
    angle_factor = get_angle_factor(inverse)
    output_stride = 1

    for _ in range(level_count):
        prime = 2
        sub_squences = [register_list[i::N//prime] for i in range(N//prime)]
        block_width = output_stride * prime
        outer_twiddle_stride = N // block_width

        base_twiddle = None
        if isinstance(twiddle_index, int):
            if twiddle_index != 0:
                base_twiddle = npc.exp_complex(1j * angle_factor * outer_twiddle_stride * twiddle_index / twiddle_N)
        else:
            resources.omega_register.real = (angle_factor * outer_twiddle_stride / twiddle_N) * twiddle_index
            resources.omega_register[:] = vc.complex_from_euler_angle(resources.omega_register.real)
            base_twiddle = resources.omega_register

        for i in range(0, N // prime):
            inner_block_offset = i % output_stride
            block_index = (i * prime) // block_width
            fixed_twiddle = npc.exp_complex(1j * angle_factor * inner_block_offset / block_width)

            _apply_combined_twiddle_to_register(
                resources=resources,
                register=sub_squences[i][1],
                base_twiddle=base_twiddle,
                fixed_twiddle=fixed_twiddle
            )
            radix_P(resources, inverse, sub_squences[i])

            sub_sequence_offset = block_index * block_width + inner_block_offset

            for j in range(prime):
                register_list[sub_sequence_offset + j * output_stride] = sub_squences[i][j]

        output_stride *= prime

    return register_list

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

    vc.comment(f"Performing a Radix-{primes} FFT on {N} registers")

    if len(primes) > 0 and all(prime == 2 for prime in primes):
        vc.comment("Fusing inter-stage and intra-stage twiddles into radix-2 decomposition levels")
        return _radix_composite_fused_power_of_two(
            resources=resources,
            inverse=inverse,
            register_list=register_list,
            level_count=len(primes),
            twiddle_index=twiddle_index,
            twiddle_N=twiddle_N
        )

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
