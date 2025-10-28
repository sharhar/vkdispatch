import vkdispatch as vd
import vkdispatch.codegen as vc

from .resources import FFTResources

import dataclasses

@dataclasses.dataclass
class MemoryOp:
    fft_offset: vc.ShaderVariable
    fft_stride: int
    fft_index: vc.ShaderVariable
    fft_size: int
    register_id: int
    register_count: int
    element_id: int
    element_count: int
    instance_id: int
    instance_count: int

def memory_reads_iterator(resources: FFTResources, stage_index: int = 0):
    resources.stage_begin(stage_index)

    index_list = list(range(resources.config.register_count))
    invocations = resources.invocations[stage_index]

    for ii, invocation in enumerate(invocations):
        resources.invocation_gaurd(stage_index, ii)

        register_indicies = index_list[invocation.register_selection]

        offset = invocation.instance_id
        stride = resources.config.N // resources.config.stages[stage_index].fft_length

        for i in range(len(register_indicies)):
            fft_index = i * stride + offset

            read_op = MemoryOp(
                fft_offset=offset,
                fft_stride=stride,
                fft_index=fft_index,
                fft_size=resources.config.N,
                register_id=register_indicies[i],
                register_count=resources.config.register_count,
                element_id=i,
                element_count=len(register_indicies),
                instance_id=ii,
                instance_count=len(invocations)
            )

            yield read_op

    resources.invocation_end(stage_index)
    resources.stage_end(stage_index)

def memory_writes_iterator(resources: FFTResources, stage_index: int = -1):
    resources.stage_begin(stage_index)

    index_list = list(range(resources.config.register_count))
    element_count = resources.config.stages[stage_index].fft_length
    invocations = resources.invocations[stage_index]

    for i in range(element_count):
        for ii, invocation in enumerate(invocations):
            resources.invocation_gaurd(stage_index, ii)

            offset = invocation.sub_sequence_offset
            stride = resources.output_strides[stage_index]

            fft_index = offset + i * stride

            register_indicies = index_list[invocation.register_selection]

            write_op = MemoryOp(
                fft_offset=offset,
                fft_stride=stride,
                fft_index=fft_index,
                fft_size=resources.config.N,
                register_id=register_indicies[i],
                register_count=resources.config.register_count,
                element_id=i,
                element_count=element_count,
                instance_id=ii,
                instance_count=len(invocations)
            )

            yield write_op

    resources.invocation_end(stage_index)
    resources.stage_end(stage_index)