import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np
#import tqdm

#from matplotlib import pyplot as plt

import sys

vd.initialize(log_level=vd.LogLevel.INFO)

print("Initializing...")

vd.make_context(devices=[0], queue_families=[[2]])

print("Context created")

shape = (512, 512)

arr = np.random.rand(shape[0], shape[1]) + 1j * np.random.rand(shape[0], shape[1])

buffer = vd.asbuffer(arr.astype(np.complex64))
buffer2 = vd.asbuffer(np.zeros(shape, dtype=np.complex64))

cmd_list = vd.CommandList()

fft_count = 20
submit_count = 100000

for _ in range(fft_count):
    vd.fft(buffer, cmd_list=cmd_list)

print("FFT commands generated")

#status_bar = tqdm.tqdm(total=fft_count * submit_count * 100)

for i in range(submit_count):
    cmd_list.submit_any(instance_count=100) #data=data_buff.tobytes())
    #status_bar.update(fft_count * 100)

    print("Commands submitted")

    #if i % 10 == 0:
    #    buffer.read()
    #    buffer2.read()

buffer.read()
buffer2.read()