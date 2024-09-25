import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np
import tqdm

from matplotlib import pyplot as plt

import sys

#vd.initialize(log_level=vd.LogLevel.INFO)

vd.make_context(max_streams=False)

shape = (256, 256)

arr = np.random.rand(shape[0], shape[1]) + 1j * np.random.rand(shape[0], shape[1])

buffer = vd.asbuffer(arr.astype(np.complex64))
buffer2 = vd.asbuffer(np.zeros(shape, dtype=np.complex64))

cmd_list = vd.CommandList()

fft_count = 2000
submit_count = 1000

for _ in range(fft_count):
    vd.fft(buffer, cmd_list=cmd_list)

status_bar = tqdm.tqdm(total=fft_count * submit_count)

for i in range(submit_count):
    cmd_list.submit(instance_count=1) #data=data_buff.tobytes())
    status_bar.update(fft_count)

    #if i % 10 == 0:
    #    buffer.read()
    #    buffer2.read()

buffer.read()
buffer2.read()