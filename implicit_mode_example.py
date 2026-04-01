import numpy as np
import math
import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abbreviations import *

N = 256
x = np.arange(N, dtype=np.float32)

signal = (
    np.sin(2 * np.pi * 5 * x / N)
    + 0.5 * np.sin(2 * np.pi * 17 * x / N)
    + 0.25 * np.sin(2 * np.pi * 40 * x / N)
).astype(np.float32).astype(np.complex64)

input_buffer = vd.asbuffer(signal.copy())
lowpass_output = vd.Buffer((N,), vd.complex64)
derivative_output = vd.Buffer((N,), vd.complex64)

omega = 2.0 * math.pi / N
shape = (N,)

with vd.fft.fft_context(shape) as ctx:
    args = ctx.declare_shader_args([
        vc.Buffer[vc.c64],  # input
        vc.Buffer[vc.c64],  # low-pass output
        vc.Buffer[vc.c64],  # derivative output
    ])

    ctx.read_from_buffer(args[0])
    ctx.execute(inverse=False)
    ctx.register_shuffle()

    dx_registers = ctx.allocate_registers("dx")

    # Branch 1: Gaussian low-pass.
    for op in ctx.reads_iter() :#vd.fft.memory_reads_iterator(ctx.resources):
        reg = ctx.registers[op.register_id]
        reg_d = dx_registers[op.register_id]

        k = op.fft_index_shifted
        
        reg_d[:] = k * omega * vc.mult_complex(reg, 1j)
        reg *= vc.exp(-(k * k) * 0.5 / (10.0 * 10.0))

    # IFFT and write low-pass output
    ctx.execute(inverse=True)
    ctx.registers.normalize()
    ctx.write_to_buffer(args[1])

    # Load spectrum again for derivative output
    ctx.registers.read_from_registers(dx_registers)

    # IFFT and write derivative output
    ctx.execute(inverse=True)
    ctx.registers.normalize()
    ctx.write_to_buffer(args[2])

kernel = ctx.get_callable()
kernel(input_buffer, lowpass_output, derivative_output)

gpu_lowpass = lowpass_output.read(0)
gpu_derivative = derivative_output.read(0)

freq = np.arange(N, dtype=np.int32)
freq[freq > N // 2] -= N
fft_signal = np.fft.fft(signal)

lowpass_weight = np.exp(-0.5 * (freq.astype(np.float32) / 10.0) ** 2).astype(np.float32)
ref_lowpass = np.fft.ifft(fft_signal * lowpass_weight).astype(np.complex64)

deriv_multiplier = 1j * (2.0 * np.pi / N) * freq.astype(np.float32)
ref_derivative = np.fft.ifft(fft_signal * deriv_multiplier).astype(np.complex64)

print("low-pass max abs err:", float(np.max(np.abs(gpu_lowpass - ref_lowpass))))
print("derivative max abs err:", float(np.max(np.abs(gpu_derivative - ref_derivative))))

# print(kernel.get_src(line_numbers=True))
