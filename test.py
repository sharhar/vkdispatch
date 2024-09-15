from traceback import print_tb
import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np
import tqdm

from matplotlib import pyplot as plt

import sys

vd.initialize(log_level=vd.LogLevel.INFO)

shape = (512, 512)

arr = np.random.rand(shape[0], shape[1]) + 1j * np.random.rand(shape[0], shape[1])

buffer = vd.asbuffer(arr.astype(np.complex64))
buffer2 = vd.asbuffer(np.zeros(shape, dtype=np.complex64))

cmd_list = vd.CommandList()


fft_count = 100
submit_count = 1000
instance_count = 10

@vc.shader(exec_size=lambda args: args.buffer.size)
def do_shader(buffer: Buff[c64], buffer2: Buff[c64], alpha: Var[f32]):
    ind = vc.global_invocation.x

    buffer2[ind] = buffer[ind] + buffer2[ind] * alpha

variables = vd.LaunchVariables()

print("Running FFTs")

for _ in range(fft_count):
    print("FFT")
    vd.fft(buffer, cmd_list=cmd_list)
    print("IFFT")
    vd.ifft(buffer, cmd_list=cmd_list)

    vd.fft(buffer, cmd_list=cmd_list)
    vd.fft(buffer, cmd_list=cmd_list)
    vd.ifft(buffer, cmd_list=cmd_list)
    vd.ifft(buffer, cmd_list=cmd_list)
    #do_shader(buffer, buffer2, variables["alpha"])

status_bar = tqdm.tqdm(total=instance_count * fft_count * 4 * submit_count)

data_buff = np.ones((instance_count, ), dtype=np.float32) * 0.5

for i in range(submit_count):
    cmd_list.submit(instance_count=instance_count) #data=data_buff.tobytes())
    status_bar.update(instance_count * fft_count * 4)

    #if i % 10 == 0:
    #    buffer.read()
    #    buffer2.read()

buffer.read()
buffer2.read()