import vkdispatch as vd
import numpy as np

vd.initialize(debug_mode=True)

N = 64

out = vd.fft.make_fft_stage(N)

print(out)

#signal = np.ones(shape=(N,)) #np.random.rand(N) # + 1j * np.random.rand(N)
#signal[1] = 0

signal = np.random.rand(N) + 1j * np.random.rand(N)
signal_fft = np.fft.fft(signal)

signal_gpu = vd.asbuffer(signal.astype(np.complex64))
out(signal_gpu, exec_size=N // 2)

print(signal_fft)
print(signal_gpu.read(0))

print(np.allclose(signal_fft, signal_gpu.read(0)))

#print(out)
