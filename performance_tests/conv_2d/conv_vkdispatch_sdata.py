import csv
import time
import conv_utils as fu
import vkdispatch as vd
import vkdispatch.codegen as vc
import numpy as np

def run_vkdispatch(config: fu.Config, fft_size: int) -> float:
    shape = config.make_shape(fft_size)
    random_data = config.make_random_data(fft_size)
    random_data_2 = config.make_random_data(fft_size)

    buffer = vd.Buffer(shape, var_type=vd.complex64)
    buffer.write(random_data)

    kernel = vd.Buffer(shape, var_type=vd.complex64)
    kernel.write(random_data_2)

    graph = vd.CommandGraph()

    @vd.map_registers([vc.c64])
    def kernel_mapping(kernel_buffer: vc.Buffer[vc.c64]):
        img_val = vc.mapping_registers()[0]
        read_register = vc.mapping_registers()[1]

        # Calculate the invocation within this FFT batch
        in_group_index = vc.local_invocation().y * vc.workgroup_size().x + vc.local_invocation().x
        out_group_index = vc.workgroup().y * vc.num_workgroups().x + vc.workgroup().x
        workgroup_index = in_group_index + out_group_index * (
            vc.workgroup_size().x * vc.workgroup_size().y
        )

        # Calculate the batch index of the FFT
        batch_index = (
            vc.mapping_index()
        ) / (
            vc.workgroup_size().x * vc.workgroup_size().y *
            vc.num_workgroups().x * vc.num_workgroups().y
        )

        # Calculate the transposed index
        transposed_index = workgroup_index + batch_index * (
            vc.workgroup_size().x * vc.workgroup_size().y *
            vc.num_workgroups().x * vc.num_workgroups().y
        )

        read_register[:] = kernel_buffer[transposed_index]
        img_val[:] = vc.mult_conj_c64(read_register, img_val)
    
    vd.fft.fft(buffer, graph=graph, disable_compute=True)
    vd.fft.convolve(buffer, kernel, axis=1, graph=graph, kernel_map=kernel_mapping, disable_compute=True)
    vd.fft.fft(buffer, graph=graph, inverse=True, disable_compute=True)

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

    output_name = f"conv_vkdispatch_sdata.csv"
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

            writer.writerow(["vkdispatch_sdata", fft_size] + rounded_data + [rounded_mean, rounded_std])
        
    print(f"Results saved to {output_name}.csv")


    