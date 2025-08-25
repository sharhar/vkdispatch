import csv
import time
import conv_padded_utils as fu
import vkdispatch as vd
import vkdispatch.codegen as vc
import numpy as np

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

    @vd.map_registers([vc.c64])
    def pad_zeros(buff: vc.Buff[vc.c64]):
        vc.if_all(
            vc.mapping_index() % buffer.shape[2] < signal_size,
            (vc.mapping_index() / buffer.shape[2]) % buffer.shape[1] < signal_size)

        vc.mapping_registers()[0][:] = buff[vc.mapping_index()]
        vc.else_statement()
        vc.mapping_registers()[0][:] = "vec2(0)"
        vc.end()
    
    vd.fft.convolve2D(buffer, kernel, graph=graph, input_map=pad_zeros)

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


    