import vkdispatch as vd
import vkdispatch.codegen as vc

import dataclasses
import enum

from typing import List
from typing import Any
from typing import Dict
from typing import Tuple

import numpy as np

vd.make_context(use_cpu=True)

class CommandType(enum.Enum):
    ADD_VALUE = 0
    SUB_VALUE = 1
    MULT_VALUE = 2
    DIV_VALUE = 3
    SIN_VALUE = 4
    COS_VALUE = 5

command_type_to_str = {
    CommandType.ADD_VALUE: "ADD",
    CommandType.SUB_VALUE: "SUB",
    CommandType.MULT_VALUE: "MULT",
    CommandType.DIV_VALUE: "DIV",
    CommandType.SIN_VALUE: "SIN",
    CommandType.COS_VALUE: "COS"
}

@dataclasses.dataclass
class ProgramCommand:
    command_type: CommandType
    value: float

@dataclasses.dataclass
class RunConfig:
    buffer_count: int
    buffer_sizes: List[int]

    program_count: int
    program_commands: List[List[ProgramCommand]]

    def __repr__(self):
        commands_repr = ""

        for commands in self.program_commands:
            commands_repr += "\n"

            for command in commands:
                command_name = command_type_to_str[command.command_type]

                commands_repr += f"        {command_name} {command.value}\n"

        return f"""RunConfig(
    buffer_count={self.buffer_count}, 
    buffer_sizes={self.buffer_sizes}, 
    program_count={self.program_count}, 
    program_commands=[{commands_repr}
])"""

def make_random_config() -> RunConfig:
    buffer_count = np.random.randint(2, 15)
    buffer_sizes = np.random.randint(100, 1000, size=buffer_count).tolist()

    program_count = np.random.randint(2, 4)
    program_commands = []

    for _ in range(program_count):
        command_count = np.random.randint(2, 15)
        commands = []

        for _ in range(command_count):
            command_type = np.random.choice(list(CommandType))
            value = np.random.uniform(-10, 10)

            commands.append(ProgramCommand(command_type, value))

        program_commands.append(commands)

    return RunConfig(
        buffer_count=buffer_count,
        buffer_sizes=buffer_sizes,
        program_count=program_count,
        program_commands=program_commands
    )

buffer_cache: Dict[int, vd.Buffer] = {}

def get_buffer(index: int, config: RunConfig) -> vd.Buffer:
    global buffer_cache
    
    if index not in buffer_cache:
        buffer_cache[index] = vd.asbuffer(
            np.zeros(
                shape=(config.buffer_sizes[index],), 
                dtype=np.float32
            )
        )

    return buffer_cache[index]

def make_source(commands: List[ProgramCommand]):
    local_size_x = vd.get_context().max_workgroup_size[0]

    header = """
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(push_constant) uniform PushConstant {
    uint exec_count;
} PC;

layout(set = 0, binding = 0) buffer Buffer0 { float data[]; } bufOut;
layout(set = 0, binding = 1) buffer Buffer1 { float data[]; } bufIn;
""" + f"""
layout(local_size_x = {local_size_x}, local_size_y = 1, local_size_z = 1) in;
""" + """
void main() {
        if(PC.exec_count <= gl_GlobalInvocationID.x) {
            return ;
        }

        uint tid = gl_GlobalInvocationID.x;

        float value = bufIn.data[tid];
"""

    ending = """
        bufOut.data[tid] = value;
}
"""

    return header + ending

program_cache: Dict[int, vd.ComputePlan] = {}

def get_program(index: int, config: RunConfig) -> vd.ComputePlan:
    global program_cache

    if index not in program_cache:
        program_cache[index] = vd.ComputePlan(
            shader_source=make_source(config.program_commands[index]),
            binding_types=[1, 1],
            push_constant_size=4,
            name=f"program_{index}"
        )

    return program_cache[index]

descriptor_set_cache: Dict[Tuple[int, int, int], vd.DescriptorSet] = {}

def get_descriptor_set(out_buffer: int, in_buffer: int, program_handle, config: RunConfig) -> vd.DescriptorSet:
    global descriptor_set_cache

    dict_key = (out_buffer, in_buffer, program_handle)

    if dict_key not in descriptor_set_cache:
        output_buffer = get_buffer(out_buffer, config)
        input_buffer = get_buffer(in_buffer, config)

        descriptor_set = vd.DescriptorSet(program_handle)
        descriptor_set.bind_buffer(output_buffer, 0)
        descriptor_set.bind_buffer(input_buffer, 1)

        descriptor_set_cache[dict_key] = descriptor_set

    return descriptor_set_cache[dict_key]

def clear_caches():
    global buffer_cache
    global program_cache
    global descriptor_set_cache

    buffer_cache.clear()
    program_cache.clear()
    descriptor_set_cache.clear()

def do_vkdispatch_command(cmd_list: vd.CommandList, out_buffer: int, in_buffer: int, program: int, config: RunConfig):    
    compute_plan = get_program(program, config)
    descriptor_set = get_descriptor_set(out_buffer, in_buffer, compute_plan._handle, config)

    cmd_list.reset()
    
    local_size = vd.get_context().max_workgroup_size[0]

    total_exec_size = min(config.buffer_sizes[out_buffer], config.buffer_sizes[in_buffer])

    block_count = (total_exec_size + local_size - 1) // local_size

    cmd_list.record_compute_plan(compute_plan, descriptor_set, [block_count, 1, 1])

    cmd_list.submit(data=np.array([total_exec_size], dtype=np.uint32).tobytes())

config = make_random_config()

print(config)


#def do_numpy_command():
#    pass

#def test_async_commands():
#    pass
