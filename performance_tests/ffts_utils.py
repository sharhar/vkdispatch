import vkdispatch as vd
from scipy import fft
import time
import torch
import numpy as np
import warp as wp
import cupy as cp

import os

from typing import Tuple

reference_list = []

def register_object(obj):
    reference_list.append(obj)

def run_torch(shape: Tuple[int, ...], axis: int, iter_count: int, iter_batch: int, warmup: int) -> float:
    # At the top of your script
    torch.backends.cuda.cufft_plan_cache.max_size = 1024  # Increase cache size

    # buffer = torch.empty(
    #     shape,
    #     dtype=torch.complex64,
    #     device='cuda'
    # )

    buffer = cp.zeros(shape, dtype=np.complex64)

    # with wp.ScopedCapture(device=wp.get_device("cuda:0")) as capture:
    #     cp.fft.fft(buffer, axis=axis)

    #graph = capture.graph
    
    for _ in range(warmup):
        cp.fft.fft(buffer, axis=axis)

    cp.cuda.Stream.null.synchronize()  # Ensure all operations are complete
    #torch.cuda.synchronize()

    gb_byte_count = 2 * np.prod(shape) * 8 / (1024 * 1024 * 1024)
    
    start_time = time.perf_counter()

    for _ in range(iter_count):
        cp.fft.fft(buffer, axis=axis)

    cp.cuda.Stream.null.synchronize()  # Ensure all operations are complete

    elapsed_time = time.perf_counter() - start_time

    return iter_count * gb_byte_count / elapsed_time


def run_vkdispatch(shape: Tuple[int, ...], axis: int, iter_count: int, iter_batch: int, warmup: int) -> float:
    buffer = vd.Buffer(shape, var_type=vd.complex64)
    output_buffer = vd.Buffer(shape, var_type=vd.complex64)
    buffer_shape = buffer.shape

    register_object(buffer)
    register_object(output_buffer)

    graph = vd.CommandGraph()

    register_object(graph)

    vd.fft.fft(
        output_buffer,
        buffer,
        buffer_shape=buffer_shape,
        graph=graph,
        axis=axis,
    )

    for _ in range(warmup):
        graph.submit(iter_batch)

    vd.queue_wait_idle()

    gb_byte_count = 2 * 8 * output_buffer.size / (1024 * 1024 * 1024)
    
    start_time = time.perf_counter()

    for _ in range(iter_count // iter_batch):
        graph.submit(iter_batch)

    vd.queue_wait_idle()

    elapsed_time = time.perf_counter() - start_time

    #buffer.destroy()
    #output_buffer.destroy()
    #graph.destroy()

    #vd.queue_wait_idle()

    # del buffer, output_buffer, graph

    return iter_count * gb_byte_count / elapsed_time

def run_vkfft(shape: Tuple[int, ...], axis: int, iter_count: int, iter_batch: int, warmup: int) -> float:
    buffer = vd.Buffer(shape, var_type=vd.complex64)
    output_buffer = vd.Buffer(shape, var_type=vd.complex64)
    buffer_shape = buffer.shape

    graph = vd.CommandGraph()

    register_object(buffer)
    register_object(output_buffer)
    register_object(graph)

    vd.vkfft.fft(
        output_buffer,
        buffer,
        buffer_shape=buffer_shape,
        graph=graph,
        axis=axis,
    )

    for _ in range(warmup):
        graph.submit(iter_batch)

    vd.queue_wait_idle()

    gb_byte_count = 2 * 8 * output_buffer.size / (1024 * 1024 * 1024)
    
    start_time = time.perf_counter()

    for _ in range(iter_count // iter_batch):
        graph.submit(iter_batch)

    vd.queue_wait_idle()

    elapsed_time = time.perf_counter() - start_time

    return iter_count * gb_byte_count / elapsed_time

def run_scipy(shape: Tuple[int, ...], axis: int, iter_count: int, iter_batch: int, warmup: int) -> float:
    buffer = np.empty(shape, dtype=np.complex64)
    output_buffer = np.empty_like(buffer)

    for _ in range(warmup):
        output_buffer = fft.fft(buffer, axis=axis, workers= os.cpu_count() - 1)

    start_time = time.perf_counter()

    for _ in range(iter_count):
        output_buffer = fft.fft(buffer, axis=axis, workers= os.cpu_count() - 1)

    elapsed_time = time.perf_counter() - start_time

    gb_byte_count = 2 * buffer.nbytes / (1024 * 1024 * 1024)

    return iter_count * gb_byte_count / elapsed_time

backends = {
    'torch': run_torch,
    'vkdispatch': run_vkdispatch,
    'vkfft': run_vkfft,
    'scipy': run_scipy
}

def run_benckmark(
    backend: str,
    shape: Tuple[int, ...],
    axis: int,
    iter_count: int,
    iter_batch: int,
    warmup: int = 10
) -> float:
    if backend not in backends:
        raise ValueError(f"Unsupported backend: {backend}")

    return backends[backend](shape, axis, iter_count, iter_batch, warmup)