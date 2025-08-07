import vkdispatch as vd
import vkdispatch.codegen as vc
import numpy as np

def calc_dft_item(out_values, in_values, temp_var, i, j, N):
    # if j is 0, copy the value
    if j == 0:
        out_values[i][:] = in_values[j]
        return

    # if i is 0, add the value
    if i == 0:
        out_values[i] += in_values[j]
        return

    # if i * j is N // 2, subtract the value
    if i * j == N // 2 and N % 2 == 0:
        out_values[i] -= in_values[j]
        return

    # Calculate the DFT item using the complex exponential if not a special case
    omega = np.exp(-2j * np.pi * i * j / N)

    temp_var.x = in_values[j].x * omega.real - in_values[j].y * omega.imag
    temp_var.y = in_values[j].x * omega.imag + in_values[j].y * omega.real

    out_values[i] += temp_var

def do_dft_N(in_values, out_values):
    N = len(out_values)

    vc.comment(f"Performing DFT on {N} complex values")
    temp_var = vc.new_vec2(0.0, var_name="temp_var")

    # These loops occur during shader generation, not at runtime
    for i in range(0, N):
        vc.comment(f"Calculating DFT for index {i}")

        for j in range(0, N):
            calc_dft_item(out_values, in_values, temp_var, i, j, N)

def make_dft_shader(N: int):
    @vd.shader()
    def demo_dft_shader(buff: vc.Buff[vc.c64]):
        vc.comment(f"Allocating varibles for DFT on stack")
        in_values = [buff[i].copy(var_name=f"in_{i}") for i in range(N)]
        out_values = [vc.new_vec2(0, var_name=f"out_{i}") for i in range(N)]

        do_dft_N(in_values, out_values)

        vc.comment("Writing output values back to buffer")

        for i in range(N):
            buff[i] = out_values[i]

    return demo_dft_shader

dft_shader = make_dft_shader(3)
print(dft_shader)