import vkdispatch as vd
import vkdispatch.codegen as vc
import tqdm
import numpy as np
from matplotlib import pyplot as plt


def transpose_kernel(
        correlation_buffer: vd.Buffer,
        image_dft_buffer: vd.Buffer,
        image_dft_buffer_transposed: vd.Buffer):
    
    @vd.map_registers([vc.c64])
    def kernel_mapping(
        kernel_buffer: vc.Buffer[vc.c64],
        kernel_transposed_buffer: vc.Buffer[vc.c64]):

        read_register = vc.mapping_registers()[1]

        # We skip batches other than the first one, since we only have one kernel
        vc.if_statement(
            vc.mapping_index() >= correlation_buffer.shape[1] * correlation_buffer.shape[2]
        )
        vc.return_statement()
        vc.end()

        # Calculate the invocation within this FFT batch
        in_group_index = vc.local_invocation().y * vc.workgroup_size().x + vc.local_invocation().x
        out_group_index = vc.workgroup().y * vc.num_workgroups().x + vc.workgroup().x
        workgroup_index = in_group_index + out_group_index * (
            vc.workgroup_size().x * vc.workgroup_size().y
        )

        # Calculate the batch index of the FFT
        batch_index = vc.mapping_index() / (
            vc.workgroup_size().x * vc.workgroup_size().y *
            vc.num_workgroups().x * vc.num_workgroups().y
        )

        # Calculate the transposed index
        transposed_index = workgroup_index + batch_index * (
            vc.workgroup_size().x * vc.workgroup_size().y *
            vc.num_workgroups().x * vc.num_workgroups().y
        )

        read_register[:] = kernel_buffer[vc.mapping_index()]
        kernel_transposed_buffer[transposed_index] = read_register
    
    vd.fft.convolve(
        correlation_buffer,
        image_dft_buffer,
        image_dft_buffer_transposed,
        kernel_map=kernel_mapping,
        axis=1
    )

def padded_cross_correlation(
        buffer: vd.Buffer,
        kernel: vd.Buffer,
        signal_shape: tuple,
        graph: vd.CommandGraph):


    # Fill input buffer with zeros where needed
    @vd.map_registers([vc.c64])
    def initial_input_mapping(input_buffer: vc.Buffer[vc.c64]):
        vc.if_statement(vc.mapping_index() % buffer.shape[2] < signal_shape[1])

        in_layer_index = vc.mapping_index() % (signal_shape[1] * buffer.shape[2])
        out_layer_index = vc.mapping_index() / (signal_shape[1] * buffer.shape[2])
        actual_index = in_layer_index + out_layer_index * (buffer.shape[1] * buffer.shape[2])

        vc.mapping_registers()[0][:] = input_buffer[actual_index]
        vc.else_statement()
        vc.mapping_registers()[0][:] = "vec2(0)"
        vc.end()

    # Remap output indicies to match the actual buffer shape
    @vd.map_registers([vc.c64])
    def initial_output_mapping(output_buffer: vc.Buffer[vc.c64]):
        in_layer_index = vc.mapping_index() % (signal_shape[1] * buffer.shape[2])
        out_layer_index = vc.mapping_index() / (signal_shape[1] * buffer.shape[2])
        actual_index = in_layer_index + out_layer_index * (buffer.shape[1] * buffer.shape[2])
        output_buffer[actual_index] = vc.mapping_registers()[0]

    # Do the first FFT on the correlation buffer accross the first axis
    vd.fft.fft(
        buffer,
        buffer,
        buffer_shape=(
            buffer.shape[0],
            signal_shape[1],
            buffer.shape[2]
        ),
        input_map=initial_input_mapping,
        output_map=initial_output_mapping,
        graph=graph
    )

    @vd.map_registers([vc.c64])
    def kernel_mapping(kernel_buffer: vc.Buffer[vc.c64]):
        img_val = vc.mapping_registers()[0]
        read_register = vc.mapping_registers()[1]

        in_group_index = vc.local_invocation().y * vc.workgroup_size().x + vc.local_invocation().x
        out_group_index = vc.workgroup().y * vc.num_workgroups().x + vc.workgroup().x
        workgroup_index = in_group_index + out_group_index * (
            vc.workgroup_size().x * vc.workgroup_size().y
        )

        batch_index = (
            vc.mapping_index() % (kernel.shape[0] * kernel.shape[1])
        ) / (
            vc.workgroup_size().x * vc.workgroup_size().y *
            vc.num_workgroups().x * vc.num_workgroups().y
        )

        transposed_index = workgroup_index + batch_index * (
            vc.workgroup_size().x * vc.workgroup_size().y *
            vc.num_workgroups().x * vc.num_workgroups().y
        )

        read_register[:] = kernel_buffer[transposed_index]
        img_val[:] = vc.mult_conj_c64(read_register, img_val)

    @vd.map_registers([vc.c64])
    def input_mapping(input_buffer: vc.Buffer[vc.c64]):
        in_layer_index = vc.mapping_index() % (
            buffer.shape[1] * buffer.shape[2]
        )

        vc.if_statement(in_layer_index / buffer.shape[2] < signal_shape[1])
        vc.mapping_registers()[0][:] = input_buffer[vc.mapping_index()]
        vc.else_statement()
        vc.mapping_registers()[0][:] = "vec2(0)"
        vc.end()

    vd.fft.convolve(
        buffer,
        buffer,
        kernel,
        #kernel_map=kernel_mapping,
        input_map=input_mapping,
        axis=1,
        graph=graph
    )

    vd.fft.ifft(buffer, graph=graph)

def do_numpy_convolution(buffer: np.ndarray, kernel: np.ndarray, signal_shape) -> np.ndarray:
    print(buffer.shape, kernel.shape, signal_shape)

    padded_buffer = np.zeros((buffer.shape[0], buffer.shape[1], buffer.shape[2]), dtype=np.complex64)
    padded_buffer[:, :signal_shape[1], :signal_shape[2]] = buffer[:, :signal_shape[1], :signal_shape[2]]

    f_buffer = np.fft.fft2(padded_buffer, axes=(-2, -1))
    convolved = np.fft.ifft2(f_buffer * np.conj(kernel), axes=(-2, -1))

    return convolved

data = np.random.rand(1, 64, 64).astype(np.complex64)
kernel_data = np.random.rand(1, 64, 64).astype(np.complex64)

buff = vd.asbuffer(data)
kernel = vd.asbuffer(kernel_data)

kernel_transposed = vd.asbuffer(np.zeros_like(kernel_data))

transpose_kernel(buff, kernel, kernel_transposed)

graph = vd.CommandGraph()
padded_cross_correlation(buff, kernel_transposed, (1, 16, 16), graph)
graph.submit()

numpy_result = do_numpy_convolution(data, kernel_data, (1, 16, 16))

plt.imshow(np.abs(buff.read(0)[0]), cmap='gray')
plt.title('Vkdispatch Result')
plt.colorbar()
plt.savefig('vkdispatch_result.png')

plt.imshow(np.abs(numpy_result[0]), cmap='gray')
plt.title('Numpy Result')
plt.colorbar()
plt.savefig('numpy_result.png')

assert np.allclose(buff.read(0), numpy_result, atol=1e-5)

#vd.fft.fft(buff, axis=0, print_shader=True)
#vd.vkfft.fft(buff, axis=0, print_shader=True)