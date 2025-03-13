import vkdispatch as vd
import vkdispatch.codegen as vc
from matplotlib import pyplot as plt
import numpy as np

vd.initialize(debug_mode=True, log_level=vd.LogLevel.INFO)

def make_random_complex_signal(shape):
    r = np.random.random(size=shape)
    i = np.random.random(size=shape)
    return (r + i * 1j).astype(np.complex64)

def make_square_signal(shape):
    signal = np.zeros(shape)
    signal[shape[0]//4:3*shape[0]//4, shape[1]//4:3*shape[1]//4] = 1
    return signal

def make_gaussian_signal(shape, dist):
    x = np.linspace(-dist, dist, shape[0])
    y = np.linspace(-dist, dist, shape[1])
    xx, yy = np.meshgrid(x, y)
    signal = np.exp(-xx**2 - yy**2)
    return signal

def cpu_convolve_2d(signal_2d, kernel_2d):
    return np.fft.irfft2(
        (np.fft.rfft2(signal_2d).astype(np.complex64) 
        * np.fft.rfft2(kernel_2d).astype(np.complex64))
    .astype(np.complex64))

side_len = 50

save_figure = True

signal = np.zeros(shape=(side_len, 2*side_len), dtype=np.float32)

signal[:] = (np.abs(make_gaussian_signal((2*side_len, side_len), 5))).astype(np.float32)

kernel_count = 3

kernels = np.zeros(shape=(kernel_count, side_len, 2*side_len), dtype=np.float32)

for i in range(kernel_count):
    kernels[i] = (np.abs(make_square_signal((side_len, 2*side_len)))).astype(np.float32) * (i + 1)

input_buffer = vd.asbuffer(signal)
kernel_buffer = vd.asrfftbuffer(kernels)

vd.create_kernel_2Dreal(kernel_buffer)

output_buffer = vd.RFFTBuffer((kernel_count, side_len, 2*side_len))

# Perform an FFT on the buffer
vd.convolve_2Dreal(output_buffer, kernel_buffer, input=input_buffer, normalize=True)

for result_index in range(0, kernel_count):
    result = output_buffer.read_real(0)[result_index]
    reference = cpu_convolve_2d(signal, kernels[result_index])

    result = np.fft.ifftshift(result)
    reference = np.fft.ifftshift(reference)

    fig, axs = plt.subplots(2, 2)

    # Plot the difference between result and reference
    axs[0, 0].imshow(result - reference)
    axs[0, 0].set_title(f'Difference')
    axs[0, 0].set_xlabel('X')
    axs[0, 0].set_ylabel('Y')
    # Add colorbar to the difference plot
    cbar = fig.colorbar(axs[0, 0].images[0], ax=axs[0, 0])
    cbar.set_label('Difference')
    
    # Plot the absolute difference between result and reference
    axs[0, 1].imshow(np.abs(result - reference))
    axs[0, 1].set_title('Absolute Difference')
    axs[0, 1].set_xlabel('X')
    axs[0, 1].set_ylabel('Y')
    # Add colorbar to the absolute difference plot
    cbar = fig.colorbar(axs[0, 1].images[0], ax=axs[0, 1])
    cbar.set_label('Absolute Difference')

    # Plot the reference
    axs[1, 0].imshow(reference)
    axs[1, 0].set_title('Reference (Kernel {})'.format(result_index))
    axs[1, 0].set_xlabel('X')
    axs[1, 0].set_ylabel('Y')
    # Add colorbar to the reference plot
    cbar = fig.colorbar(axs[1, 0].images[0], ax=axs[1, 0])
    cbar.set_label('Reference')

    # Plot the result
    axs[1, 1].imshow(result)
    axs[1, 1].set_title(f'Result {result_index}')
    axs[1, 1].set_xlabel('X')
    axs[1, 1].set_ylabel('Y')
    # Add colorbar to the result plot
    cbar = fig.colorbar(axs[1, 1].images[0], ax=axs[1, 1])
    cbar.set_label('Result')

    if save_figure:
        device_name = vd.get_context().device_infos[0].device_name.replace(' ', '_')

        import datetime
        current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        result_filename = f'convolution_test_{device_name}_{current_date}_{result_index}.png'

        plt.savefig(result_filename)
    else:
        plt.show()