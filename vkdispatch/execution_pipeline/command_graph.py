from typing import Any
from typing import Callable
from typing import List
from typing import Dict
from typing import Union
from typing import Tuple
from typing import Optional

import uuid

import numpy as np

import vkdispatch as vd
import vkdispatch.codegen as vc

from .buffer_builder import BufferUsage
from .buffer_builder import BufferBuilder

import dataclasses

@dataclasses.dataclass
class BufferBindInfo:
    """A dataclass to hold information about a buffer binding."""
    buffer: vd.Buffer
    binding: int
    shape_name: str
    read_access: bool
    write_access: bool

@dataclasses.dataclass
class ImageBindInfo:
    """A dataclass to hold information about an image binding."""
    sampler: vd.Sampler
    binding: int
    read_access: bool
    write_access: bool

class CommandGraph(vd.CommandList):
    """TODO: Docstring"""

    _reset_on_submit: bool
    submit_on_record: bool
    
    buffers_valid: bool
    
    pc_builder: BufferBuilder
    pc_values: Dict[Tuple[str, str], Any]

    uniform_builder: BufferBuilder
    uniform_values: Dict[Tuple[str, str], Any]
    uniform_bindings: Any

    uniform_constants_size: int
    uniform_constants_buffer: vd.Buffer

    uniform_descriptors: List[Tuple[vd.DescriptorSet, int, int]]

    name_to_pc_key_dict: Dict[str, List[Tuple[str, str]]]
    queued_pc_values: Dict[Tuple[str, str], Any]

    def __init__(self, reset_on_submit: bool = False, submit_on_record: bool = False) -> None:
        super().__init__()

        self.buffers_valid = False
        
        self.pc_builder = BufferBuilder(usage=BufferUsage.PUSH_CONSTANT)
        self.uniform_builder = BufferBuilder(usage=BufferUsage.UNIFORM_BUFFER)

        self.pc_values = {}
        self.uniform_values = {}
        self.name_to_pc_key_dict = {}
        self.queued_pc_values = {}

        self.uniform_descriptors = []

        self._reset_on_submit = reset_on_submit
        self.submit_on_record = submit_on_record

        self.uniform_constants_size = 0
        self.uniform_constants_buffer = vd.Buffer(shape=(4096,), var_type=vd.uint32) # Create a base static constants buffer at size 4k bytes

    def reset(self) -> None:
        """Reset the command graph by clearing the push constant buffer and descriptor
        set lists.
        """
        super().reset()

        self.pc_builder.reset()
        self.uniform_builder.reset()

        self.pc_values = {}
        self.uniform_values = {}
        self.name_to_pc_key_dict = {}
        self.queued_pc_values = {}

        self.uniform_descriptors = []
        self.buffers_valid = False
    
    def bind_var(self, name: str):
        def register_var(key: Tuple[str, str]):
            if not name in self.name_to_pc_key_dict.keys():
                self.name_to_pc_key_dict[name] = []

            self.name_to_pc_key_dict[name].append(key)

        return register_var
    
    def set_var(self, name: str, value: Any):
        if name not in self.name_to_pc_key_dict.keys():
            raise ValueError("Variable not bound!")
        
        for key in self.name_to_pc_key_dict[name]:
            self.queued_pc_values[key] = value
    
    def record_shader(self, 
                      plan: vd.ComputePlan,
                      shader_description: vc.ShaderDescription, 
                      exec_limits: Tuple[int, int, int], 
                      blocks: Tuple[int, int, int],
                      bound_buffers: List[BufferBindInfo],
                      bound_samplers: List[ImageBindInfo],
                      uniform_values: Dict[str, Any] = {},
                      pc_values: Dict[str, Any] = {},
                      shader_uuid: str = None
                    ) -> None:
        descriptor_set = vd.DescriptorSet(plan)

        if shader_uuid is None:
            shader_uuid = shader_description.name + "_" + str(uuid.uuid4())

        if len(shader_description.pc_structure) != 0:
            self.pc_builder.register_struct(shader_uuid, shader_description.pc_structure)
        
        uniform_offset, uniform_range = self.uniform_builder.register_struct(shader_uuid, shader_description.uniform_structure)

        self.uniform_descriptors.append((descriptor_set, uniform_offset, uniform_range))

        self.uniform_values[(shader_uuid, shader_description.exec_count_name)] = [exec_limits[0], exec_limits[1], exec_limits[2], 0]

        for buffer_bind_info in bound_buffers:
            descriptor_set.bind_buffer(
                buffer=buffer_bind_info.buffer,
                binding=buffer_bind_info.binding,
                read_access=buffer_bind_info.read_access,
                write_access=buffer_bind_info.write_access,
            )
            
            self.uniform_values[(shader_uuid, buffer_bind_info.shape_name)] = buffer_bind_info.buffer.shader_shape
        
        for sampler_bind_info in bound_samplers:
            descriptor_set.bind_sampler(
                sampler_bind_info.sampler,
                sampler_bind_info.binding,
                read_access=sampler_bind_info.read_access,
                write_access=sampler_bind_info.write_access
            )

        for key, value in uniform_values.items():
            self.uniform_values[(shader_uuid, key)] = value
        
        for key, value in pc_values.items():
            self.pc_values[(shader_uuid, key)] = value

        super().record_compute_plan(plan, descriptor_set, blocks)

        self.buffers_valid = False

        if self.submit_on_record:
            self.submit()

    def submit(self, instance_count: int = None, queue_index: int = -2) -> None:
        """Submit the command list to the specified device with additional data to
        append to the front of the command list.
        
        Parameters:
        device_index (int): The device index to submit the command list to.
                Default is 0.
        data (bytes): The additional data to append to the front of the command list.
        """

        if instance_count is None:
            instance_count = 1
        
        if len(self.pc_builder.element_map) > 0 and (
                self.pc_builder.instance_count != instance_count or not self.buffers_valid
            ):

            self.pc_builder.prepare(instance_count)

            for key, value in self.pc_values.items():
                self.pc_builder[key] = value

        if len(self.uniform_builder.element_map) > 0 and not self.buffers_valid:

            self.uniform_builder.prepare(1)

            for key, value in self.uniform_values.items():
                self.uniform_builder[key] = value
            
            for descriptor_set, offset, size in self.uniform_descriptors:
                descriptor_set.bind_buffer(self.uniform_constants_buffer, 0, offset, size, True, write_access=False)

            self.uniform_constants_buffer.write(self.uniform_builder.tobytes())

        if not self.buffers_valid:
            self.buffers_valid = True

        for key, val in self.queued_pc_values.items():
            self.pc_builder[key] = val
        
        my_data = None

        if len(self.pc_builder.element_map) > 0:
            my_data = self.pc_builder.tobytes()

        super().submit(data=my_data, queue_index=queue_index, instance_count=instance_count)

        if self._reset_on_submit:
            self.reset()
    
    def submit_any(self, instance_count: int = None) -> None:
        self.submit(instance_count=instance_count, queue_index=-1)

__default_graph = None
__custom_graph = None

def default_graph() -> CommandGraph:
    global __default_graph

    if __default_graph is None:
        __default_graph = CommandGraph(reset_on_submit=True, submit_on_record=True)

    return __default_graph

def global_graph() -> CommandGraph:
    global __custom_graph

    if __custom_graph is not None:
        return __custom_graph

    return default_graph()

def set_global_graph(graph: CommandGraph = None) -> CommandGraph:
    global __custom_graph
    old_value = __custom_graph
    __custom_graph = graph 
    return old_value