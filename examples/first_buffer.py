import vkdispatch as vd
import numpy as np

# Create a simple numpy array
cpu_data = np.arange(16, dtype=np.int32)
print(f"Original CPU data: {cpu_data}")

# Create a GPU buffer
gpu_buffer = vd.asbuffer(cpu_data)

# Read data back from GPU to CPU to verify
downloaded_data = gpu_buffer.read(0)
print(f"Data downloaded from GPU: {downloaded_data.flatten()}")

