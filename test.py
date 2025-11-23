import vkdispatch as vd
import vkdispatch.codegen as vc
import numpy as np

def calc(reg_out, reg_in, phase, N):
  # if phase is 0, add the input
  if phase == 0:
    reg_out += reg_in
    return

  # if phase is 180°, subtract the input
  if phase == N // 2 and N % 2 == 0:
    reg_out -= reg_in
    return

  # Else, use complex multiplication
  w = np.exp(-2j*np.pi*phase/N)
  reg_out += vc.mult_complex(reg_in, w)

def dft(values):
  N = len(values)
  vc.comment(f"DFT on {N} values")
  outputs = []
  for i in range(0, N):
    vc.comment(f"Calc Output {i}")
    out = vc.to_complex(0)
    out = out.to_register(f"out{i}")
    for j in range(0, N):
      calc(out, values[j], i * j, N)
    outputs.append(out)
  return outputs

def make_dft_shader(N: int):
  @vd.shader()
  def dft_shader(
      buff: vc.Buff[vc.c64]):
    vc.comment("Read Input")
    values = [
      buff[i].to_register(f"in{i}")
      for i in range(N)
    ]
    
    output = dft(values)

    vc.comment("Write output")
    for i in range(N):
      buff[i] = output[i]
          
  return dft_shader

dft_shader_2 = make_dft_shader(2)
dft_shader_3 = make_dft_shader(3)

print("DFT Shader 2:")
print(dft_shader_2)

print("DFT Shader 3:")
print(dft_shader_3)