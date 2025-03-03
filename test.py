import numpy as np
import matplotlib.pyplot as plt
import vkdispatch as vd
import vkdispatch.codegen as vc

import tqdm

vd.initialize(debug_mode=True)

signal = np.ones((4096, 4096), dtype=np.float32)
#signal[:, 64:128] = 0

signal_buffer = vd.asrfftbuffer(signal)

cmd_stream = vd.CommandStream()

#vd.rfft(signal_buffer, cmd_stream=cmd_stream, padding=[(0, 0), (0, 0), (0, 0)])
vd.rfft(signal_buffer, cmd_stream=cmd_stream, padding=[(0, 0), (0, 0), (256, 4096)])
vd.irfft(signal_buffer, cmd_stream=cmd_stream)

for _ in tqdm.tqdm(range(100)):
    cmd_stream.submit(instance_count=160)

#fft_singnal = signal_buffer.read_fourier(0)

#print(fft_singnal.shape)

#real_signal = np.fft.irfft2(fft_singnal[0])


#real_buffer = signal_buffer.read_real(0)

exit()

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].imshow(np.abs(fft_singnal[0]))
axs[0, 0].set_title('Abs FFT')

axs[0, 1].imshow(np.abs(np.fft.ifft(fft_singnal[0], axis=0)))
axs[0, 1].set_title('Angle FFT')

axs[1, 0].imshow(np.abs(real_signal))
axs[1, 0].set_title('Abs real')

axs[1, 1].imshow(real_buffer[0])
axs[1, 1].set_title('angle real')

for ax in axs.flat:
    ax.label_outer()

plt.show()