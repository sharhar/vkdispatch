import csv
import time
import conv_padded_utils as fu
import vkdispatch as vd
import vkdispatch.codegen as vc
import numpy as np

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

    # Again, we skip reading the zero-padded values from the input
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
        input_map=input_mapping,
        axis=1,
        graph=graph
    )

    vd.fft.ifft(buffer, graph=graph)


def run_vkdispatch(config: fu.Config, fft_size: int) -> float:
    shape = config.make_shape(fft_size)
    random_data = config.make_random_data(fft_size)
    random_data_2 = config.make_random_data(fft_size)

    buffer = vd.Buffer(shape, var_type=vd.complex64)
    buffer.write(random_data)

    kernel = vd.Buffer(shape[1:], var_type=vd.complex64)
    kernel.write(random_data_2[0])

    graph = vd.CommandGraph()

    signal_size = fft_size // config.signal_factor

    padded_cross_correlation(buffer, kernel, (signal_size, signal_size), graph)

    for _ in range(config.warmup):
        graph.submit(config.iter_batch)

    vd.queue_wait_idle()

    gb_byte_count = 11 * 8 * buffer.size / (1024 * 1024 * 1024)
    
    start_time = time.perf_counter()

    for _ in range(config.iter_count // config.iter_batch):
        graph.submit(config.iter_batch)

    vd.queue_wait_idle()

    elapsed_time = time.perf_counter() - start_time

    buffer.destroy()
    graph.destroy()
    vd.fft.cache_clear()

    time.sleep(1)

    vd.queue_wait_idle()    

    return config.iter_count * gb_byte_count / elapsed_time

if __name__ == "__main__":
    config = fu.parse_args()
    fft_sizes = fu.get_fft_sizes()

    output_name = f"conv_padded_vkdispatch.csv"
    with open(output_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Backend', 'FFT Size'] + [f'Run {i + 1} (GB/s)' for i in range(config.run_count)] + ['Mean', 'Std Dev'])
        
        for fft_size in fft_sizes:
            rates = []

            for _ in range(config.run_count):
                gb_per_second = run_vkdispatch(config, fft_size)
                print(f"FFT Size: {fft_size}, Throughput: {gb_per_second:.2f} GB/s")
                rates.append(gb_per_second)

            rounded_data = [round(rate, 2) for rate in rates]
            rounded_mean = round(np.mean(rates), 2)
            rounded_std = round(np.std(rates), 2)

            writer.writerow(["vkdispatch", fft_size] + rounded_data + [rounded_mean, rounded_std])
        
    print(f"Results saved to {output_name}.csv")


    