import torch
import tqdm
import numpy as np
import vkdispatch as vd
import cupy as cp

print("Testing FFT performance with vkdispatch, cupy, and torch...")

# print("Cupy FFT...")
# arr = cp.array(np.zeros((10, 4096, 4096)))

# status_bar = tqdm.tqdm(total=10000)

# for i in range(1000):
#     arr = cp.fft.fft2(arr)
#     arr = cp.fft.ifft2(arr)

#     status_bar.update(10)

# reslt = cp.asnumpy(arr)

# status_bar.close()

sync_buff = vd.Buffer((10,), vd.float32)

print("VkDispatch FFT...")
buff = vd.Buffer((65535, 256, 16), vd.complex64)

cmd_stream = vd.CommandStream()

vd.fft.fft(buff, cmd_stream=cmd_stream, axis=2)
#vd.fft.ifft2(buff, cmd_stream=cmd_stream)

status_bar = tqdm.tqdm(total=1000)

for i in range(100):
    cmd_stream.submit(10)
    status_bar.update(10)

result = sync_buff.read(0)

status_bar.close()

exit()

print("Torch FFT...")
tensor = torch.zeros((10, 4096, 4096), dtype=torch.complex64)

tensor = tensor.cuda()

status_bar = tqdm.tqdm(total=10000)

for i in range(1000):
    tensor = torch.fft.fft2(tensor)
    tensor = torch.fft.ifft2(tensor)

    status_bar.update(10)

reslt = tensor.cpu().numpy()

status_bar.close()