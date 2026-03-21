import vkdispatch as vd

from vkdispatch.base.command_list import CommandList
from vkdispatch.base.compute_plan import ComputePlan
from vkdispatch.base.descriptor_set import DescriptorSet

import numpy as np

def load_shader(path: str) -> ComputePlan:
    shader_source = open(path, 'r').read()

    return ComputePlan(
        shader_source=shader_source,
        binding_type_list=[1, 1, 1],
        pc_size=0,
        shader_name=f"shader_{path.split('/')[-1].split('.')[0]}"
    )

def make_descriptor(plan: ComputePlan, out_buff: vd.Buffer, in_buff: vd.Buffer, kern_buff: vd.Buffer):
    descriptor_set = DescriptorSet(plan)

    descriptor_set.bind_buffer(out_buff, 0)
    descriptor_set.bind_buffer(in_buff, 1)
    descriptor_set.bind_buffer(kern_buff, 2)

    return descriptor_set

def numpy_convolution(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.fft.ifft(
        np.fft.fft(signal, axis=1).astype(np.complex64)
        *
        kernel.conjugate(),
        axis=1
    )

BUFF_SHAPE = (4, 512, 257)

np.random.seed(1337)

in_data = (np.random.rand(*BUFF_SHAPE) + 1j * np.random.rand(*BUFF_SHAPE)).astype(np.complex64)
kern_data = (np.random.rand(*BUFF_SHAPE) + 1j * np.random.rand(*BUFF_SHAPE)).astype(np.complex64)

reference_result_data = numpy_convolution(in_data, kern_data[0])

out_buff = vd.buffer_c64(BUFF_SHAPE)
in_buff = vd.buffer_c64(BUFF_SHAPE)
kern_buff = vd.buffer_c64(BUFF_SHAPE)

in_buff.write(in_data)
kern_buff.write(kern_data)

block_count = (1028, 32, 1)

plan_bad = load_shader("conv_bad.comp")
plan_good = load_shader("conv_good.comp")

cmd_list_bad = CommandList()

cmd_list_bad.record_compute_plan(
    plan_bad,
    make_descriptor(plan_bad, out_buff, in_buff, kern_buff),
    block_count
)

cmd_list_bad.submit(instance_count=1)

result_data_bad = out_buff.read(0)

cmd_list_good = CommandList()

cmd_list_good.record_compute_plan(
    plan_good,
    make_descriptor(plan_good, out_buff, in_buff, kern_buff),
    block_count
)

cmd_list_good.submit(instance_count=1)

result_data_good = out_buff.read(0)

for i in range(BUFF_SHAPE[0]):
    np.save(f"result_bad_{i}.npy", result_data_bad[i])
    np.save(f"result_good_{i}.npy", result_data_good[i])
    np.save(f"reference_result_{i}.npy", reference_result_data[i])
    np.save(f"diff_bad_{i}.npy", result_data_bad[i] - reference_result_data[i])
    np.save(f"diff_good_{i}.npy", result_data_good[i] - reference_result_data[i])
    np.save(f"diff_{i}.npy", result_data_good[i] - result_data_bad[i])

assert np.allclose(result_data_good, result_data_bad, atol=1e-3)
