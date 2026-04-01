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

    for read_op in ctx.reads_iter():
        read_op.read_from_buffer(args[0])

    ctx.execute(inverse=False)
    ctx.register_shuffle()

    derivative_registers = ctx.allocate_registers("derivative")

    # Branch 1: Gaussian low-pass.
    for reg_op in vd.fft.memory_reads_iterator(ctx.resources):
        reg = ctx.registers[reg_op.register_id]
        reg_d = derivative_registers[reg_op.register_id]

        k = vc.new_int_register()
        k[:] = reg_op.fft_index.to_dtype(vc.i32)
        with vc.if_block(k > (reg_op.fft_size // 2)):
            k[:] = k - reg_op.fft_size
        
        # Calculate derivative
        reg_d[:]= vc.mult_complex(reg, 1j * omega * k) # vc.to_complex(0, omega * k))
        #reg_d.real = -omega * k * reg.imag
        #reg_d.imag = omega * k * reg.real

        # Do low-pass filter
        reg *= vc.exp(-(k * k) * 0.5 / (10.0 * 10.0))

    # IFFT and write low-pass output
    ctx.execute(inverse=True)
    ctx.registers.normalize()

    for write_op in ctx.writes_iter():
        write_op.write_to_buffer(args[1])

    # Load spectrum again for derivative output
    ctx.registers.read_from_registers(derivative_registers)

    # IFFT and write derivative output
    ctx.execute(inverse=True)
    ctx.registers.normalize()

    for write_op in ctx.writes_iter():
        write_op.write_to_buffer(args[2])

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
