#!/usr/bin/env python3
"""
CUDA Kernel Launch Overhead Benchmark
Compares PyCUDA, Warp-lang, and Numba kernel launch performance
"""

import time
import numpy as np
import statistics
import gc
import sys
import os
from typing import List, Tuple, Optional
import subprocess
import sys
import tempfile
import os
from matplotlib import pyplot as plt

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray
    PYCUDA_AVAILABLE = True
except ImportError:
    print("PyCUDA not available")
    PYCUDA_AVAILABLE = False

try:
    import warp as wp
    WARP_AVAILABLE = True
except ImportError:
    print("Warp-lang not available")
    WARP_AVAILABLE = False


import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

vd.initialize(debug_mode=False, log_level=vd.LogLevel.ERROR)

class KernelLaunchBenchmark:
    def __init__(self, array_size: int = 1024, num_launches: int = 10000, batch_size: int = 100):
        self.array_size = array_size
        self.num_launches = num_launches
        self.results = {}
        
        assert self.num_launches % batch_size == 0, "num_launches must be divisible by batch_size"

        # Create test data
        self.host_data = np.ones(array_size, dtype=np.float32)

        self.params_data = np.random.rand(num_launches).astype(np.float32)

        self.batch_size = batch_size

    def benchmark_vkdispatch(self) -> float:
        @vd.shader("data.size", enable_exec_bounds=False)
        def simple_add_shader(data: Buff[f32], param: Var[f32]):
            i = vc.global_invocation().x
            data[i] = param

        cmd_stream = vd.CommandStream()

        gpu_data = vd.asbuffer(self.host_data)
       
        simple_add_shader(gpu_data, cmd_stream.bind_var("param"), cmd_stream=cmd_stream)

        for _ in range(100):
            cmd_stream.set_var("param", 0.0)
            cmd_stream.submit()

        gpu_data.read(0)

        num_graph_launches = self.num_launches // self.batch_size

        start_time = time.perf_counter()
        for i in range(num_graph_launches):
            cmd_stream.set_var("param", self.params_data[i*self.batch_size:(i+1)*self.batch_size])
            cmd_stream.submit_any(instance_count=self.batch_size)
        gpu_data.read(0)
        end_time = time.perf_counter()

        return end_time - start_time
        
    def benchmark_pycuda(self) -> float:
        """Benchmark PyCUDA kernel launches using stream batching"""
        if not PYCUDA_AVAILABLE:
            return 0.0
            
        # Simple kernel that adds 1 to each element
        kernel_code = """
        __global__ void simple_add(float* data, float* params, int* indicies, int stream_index, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] = params[indicies[stream_index]];

                if (idx == 0) {
                    indicies[stream_index] += 1;
                }
            }
        }
        """

        num_batches = self.num_launches // self.batch_size
        num_streams = min(4, num_batches)  # Use up to 4 streams
        
        # Compile kernel
        mod = SourceModule(kernel_code)
        kernel_func = mod.get_function("simple_add")
        
        # Allocate GPU memory
        gpu_data = gpuarray.to_gpu(self.host_data.copy())
        param_index_data = gpuarray.to_gpu(np.zeros(shape=(num_streams,), dtype=np.int32))
        param_gpu_data = gpuarray.to_gpu(self.params_data.copy())
        
        # Calculate grid/block dimensions
        block_size = 256
        grid_size = (self.array_size + block_size - 1) // block_size
        
        # Warm-up with regular launches
        for _ in range(100):
            kernel_func(gpu_data, np.int32(self.array_size), 
                    block=(block_size, 1, 1), grid=(grid_size, 1))
        cuda.Context.synchronize()
        
        # Use multiple streams for better batching
        
        streams = [cuda.Stream() for _ in range(num_streams)]
        
        # Benchmark - launch kernels in batches across multiple streams
        start_time = time.perf_counter()
        
        for batch in range(num_batches):
            stream = streams[batch % num_streams]
            # Launch batch_size kernels on this stream without sync
            for _ in range(self.batch_size):
                kernel_func(gpu_data,
                            param_gpu_data,
                            param_index_data,
                            np.int32(batch % num_streams),
                            np.int32(self.array_size),
                        block=(block_size, 1, 1), grid=(grid_size, 1), stream=stream)
        
        # Synchronize all streams
        for stream in streams:
            stream.synchronize()
        
        end_time = time.perf_counter()
        
        # Cleanup
        for stream in streams:
            del stream
        gpu_data.gpudata.free()
        del kernel_func, mod, gpu_data
        gc.collect()
        
        return end_time - start_time

    def benchmark_warp(self) -> float:
        """Benchmark Warp kernel launches using CUDA graphs"""
        if not WARP_AVAILABLE:
            return 0.0
            
        # Define simple Warp kernel
        @wp.kernel
        def simple_add_kernel(data: wp.array(dtype=float), params: wp.array(dtype=float), index: wp.array(dtype=int)):
            i = wp.tid()
            if i < data.shape[0]:
                data[i] = params[index[0]]

                if i == 0:
                    index[0] += 1

        # Allocate GPU memory
        gpu_data = wp.array(self.host_data.copy(), dtype=wp.float32, device="cuda:0")
        gpu_param_data = wp.array(self.params_data.copy(), dtype=wp.float32, device="cuda:0")
        gpu_index_data = wp.array(np.zeros(shape=(1,), dtype=np.int32), device="cuda:0")
        
        # Warm-up with regular launches
        for _ in range(100):
            wp.launch(simple_add_kernel,
                    dim=self.array_size, 
                    inputs=[gpu_data, gpu_param_data, gpu_index_data],
                    device="cuda:0")
        wp.synchronize_device("cuda:0")
        
        # Create CUDA graph using Warp's capture mechanism
        num_graph_launches = self.num_launches // self.batch_size
        
        # Capture graph - batch multiple kernel launches
        with wp.ScopedCapture(device="cuda:0") as capture:
            for _ in range(self.batch_size):
                wp.launch(simple_add_kernel, dim=self.array_size,
                        inputs=[gpu_data, gpu_param_data, gpu_index_data], device="cuda:0")
        
        # Get the captured graph
        graph = capture.graph
        
        # Benchmark - launch the graph multiple times
        start_time = time.perf_counter()
        for _ in range(num_graph_launches):
            wp.capture_launch(graph)
        wp.synchronize_device("cuda:0")
        end_time = time.perf_counter()
        
        # Cleanup
        del gpu_data, graph
        gc.collect()
        
        return end_time - start_time
    
    def run_single_library_trial(self, library: str) -> Optional[float]:
        """Run a single trial for one library"""
        if library == 'pycuda' and PYCUDA_AVAILABLE:
            return self.benchmark_pycuda()
        elif library == 'warp' and WARP_AVAILABLE:
            return self.benchmark_warp()
        elif library == 'vkdispatch':
            return self.benchmark_vkdispatch()
        return None
    
    def run_multiple_trials(self, num_trials: int = 5) -> dict:
        """Run multiple trials and calculate statistics"""
        # IMPORTANT: Run Numba first to ensure it gets primary context
        results = {
            'pycuda': [],
            'warp': [],
            'vkdispatch': [],
        }
        
        print(f"Running {num_trials} trials with {self.num_launches} launches each...")
        print(f"Array size: {self.array_size}")
        print("-" * 60)
        
        # Run each library separately to avoid context conflicts
        for library in results.keys():
            if not self._is_library_available(library):
                continue
                
            print(f"\nBenchmarking {library.upper()}:")
            library_times = []
            
            for trial in range(num_trials):
                print(f"  Trial {trial + 1}/{num_trials}...", end=" ")
                
                try:
                    elapsed = self.run_single_library_trial(library)
                    if elapsed is not None:
                        library_times.append(elapsed)
                        launches_per_sec = self.num_launches / elapsed
                        print(f"{elapsed:.4f}s ({launches_per_sec:,.0f} launches/sec)")
                    else:
                        print("Failed")
                except Exception as e:
                    print(f"Error: {e}")
            
            results[library] = library_times
        
        return results
    
    def _is_library_available(self, library: str) -> bool:
        """Check if library is available"""
        if library == 'pycuda':
            return PYCUDA_AVAILABLE
        elif library == 'warp':
            return WARP_AVAILABLE
        elif library == 'vkdispatch':
            return True
        return False
    
    def print_summary(self, results: dict):
        """Print summary statistics"""
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(f"{'Library':<12} {'Mean (s)':<12} {'Std Dev':<12} {'Launches/sec':<15}")
        print("-" * 60)
        
        valid_results = {}
        for lib_name, times in results.items():
            if not times:
                continue
                
            mean_time = statistics.mean(times)
            std_dev = statistics.stdev(times) if len(times) > 1 else 0
            mean_launches_per_sec = self.num_launches / mean_time
            valid_results[lib_name] = mean_time
            
            print(f"{lib_name.capitalize():<12} {mean_time:<12.4f} "
                  f"{std_dev:<12.4f} {mean_launches_per_sec:<15,.0f}")
        
        print("-" * 60)
        
        # Find fastest
        if valid_results:
            fastest_lib = min(valid_results.keys(), key=lambda x: valid_results[x])
            fastest_rate = self.num_launches / valid_results[fastest_lib]
            print(f"Fastest: {fastest_lib.capitalize()} at {fastest_rate:,.0f} launches/sec")


def get_device_info():
    """Get CUDA device information"""
    try:
        if PYCUDA_AVAILABLE:
            device = cuda.Device(0)
            print("CUDA Device:", device.name())
        else:
            print("No CUDA device info available")
    except Exception as e:
        print(f"Could not get CUDA device info: {e}")


def do_comparison_benchmark(
        array_size: int,
        num_launches: int,
        batch_size: int,
        num_trials: int = 5):
    benchmark = KernelLaunchBenchmark(array_size=array_size, 
                                        num_launches=num_launches,
                                        batch_size=batch_size)
        
    results_dict = benchmark.run_multiple_trials(num_trials=num_trials)

    means_dict = {
        lib: np.mean(times) if times else 0.0
        for lib, times in results_dict.items()
    }

    std_dict = {
        lib: np.std(times) if len(times) > 1 else 0.0
        for lib, times in results_dict.items()
    }

    return means_dict, std_dict

def main():
    print("CUDA Kernel Launch Overhead Benchmark")
    print("=" * 60)
    
    get_device_info()
    print()
    
    # Initialize Warp only after Numba is set up
    if WARP_AVAILABLE:
        wp.init()

    # Configuration
    array_size = 10  # Small array to minimize GPU work
    num_launches = 50000
    num_trials = 5
    batch_sizes = [1, 5, 10, 50, 200, 1000]  # Number of kernel launches per graph

    means = {
        'pycuda': [],
        'warp': [],
        'vkdispatch': [],
    }

    stds = {
        'pycuda': [],
        'warp': [],
        'vkdispatch': [],
    }

    for batch_size in batch_sizes:
        print(f"Benchmarking with data size: {array_size * 4 / (1024 * 1024)} MB")
        run_results = do_comparison_benchmark(
            array_size=array_size,
            num_launches=num_launches,
            batch_size=batch_size,
            num_trials=num_trials
        )

        for lib, times in run_results[0].items():
            means[lib].append(times)

        for lib, times in run_results[1].items():
            stds[lib].append(times)

    # Plotting results
    plt.figure(figsize=(12, 6))
    for lib in means.keys():
        plt.errorbar(batch_sizes, means[lib], yerr=stds[lib], label=lib.upper(), marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Batch Size (Number of Kernel Launches per Graph)')
    plt.ylabel('Time (seconds)')
    plt.title('Param Streaming Overhead Benchmark')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('param_streaming_benchmark.png')
    plt.show()



if __name__ == "__main__":
    main()