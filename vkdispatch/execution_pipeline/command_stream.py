from typing import Any
from typing import Callable
from typing import List
from typing import Dict
from typing import Union
from typing import Tuple

import numpy as np

import vkdispatch as vd
import vkdispatch.codegen as vc

from .buffer_builder import BufferUsage
from .buffer_builder import BufferBuilder

class CommandStream(vd.CommandList):
    """TODO: Docstring"""

    _reset_on_submit: bool
    submit_on_record: bool
    
    pc_builder: BufferBuilder
    pc_values: Dict[Tuple[str, str], Any]

    uniform_builder: BufferBuilder
    uniform_values: Dict[Tuple[str, str], Any]
    uniform_bindings: Any

    uniform_constants_size: int
    uniform_constants_valid: bool
    uniform_constants_buffer: vd.Buffer

    uniform_descriptors: List[Tuple[vd.DescriptorSet, int, int]]

    def __init__(self, reset_on_submit: bool = False, submit_on_record: bool = False) -> None:
        super().__init__()

        self.pc_builder = BufferBuilder(usage=BufferUsage.PUSH_CONSTANT)
        self.uniform_builder = BufferBuilder(usage=BufferUsage.UNIFORM_BUFFER)

        self.pc_values = {}
        self.uniform_values = {}

        self.uniform_descriptors = []

        self._reset_on_submit = reset_on_submit
        self.submit_on_record = submit_on_record

        self.uniform_constants_size = 0
        self.uniform_constants_valid = False
        self.uniform_constants_buffer = vd.Buffer(shape=(4096,), var_type=vd.uint32) # Create a base static constants buffer at size 4k bytes

    def __del__(self) -> None:
        pass  # vkdispatch_native.command_list_destroy(self._handle)

    def reset(self) -> None:
        """Reset the command stream by clearing the push constant buffer and descriptor
        set lists.
        """

        super().reset()

        self.pc_builder.reset()
        self.uniform_builder.reset()

        self.pc_values = {}
        self.uniform_values = {}

        self.uniform_descriptors = []
        self.uniform_constants_valid = False

        super().reset()
    
    def record_shader(self, 
                      plan: vd.ComputePlan,
                      shader_description: vc.ShaderDescription, 
                      exec_limits: Tuple[int, int, int], 
                      blocks: Tuple[int, int, int],
                      bound_buffers: List[Tuple[vd.Buffer, int, str]],
                      bound_images: List[Tuple[vd.Image, int]],
                      uniform_values: Dict[str, Any] = {},
                      pc_values: Dict[str, Any] = {}
                    ) -> None:
        descriptor_set = vd.DescriptorSet(plan._handle)

        if len(shader_description.pc_structure) != 0:
            self.pc_builder.register_struct(shader_description.name, shader_description.pc_structure)
        
        uniform_offset, uniform_range = self.uniform_builder.register_struct(shader_description.name, shader_description.uniform_structure)

        self.uniform_descriptors.append((descriptor_set, uniform_offset, uniform_range))

        self.uniform_values[(shader_description.name, shader_description.exec_count_name)] = [exec_limits[0], exec_limits[1], exec_limits[2], 0]

        for buffer, binding, shape_name in bound_buffers:
            descriptor_set.bind_buffer(buffer, binding)
            self.uniform_values[(shader_description.name, shape_name)] = buffer.shader_shape
        
        for image, binding in bound_images:
            descriptor_set.bind_image(image, binding)

        for key, value in uniform_values.items():
            self.uniform_values[(shader_description.name, key)] = value
        
        for key, value in pc_values.items():
            self.pc_values[(shader_description.name, key)] = value

        super().record_compute_plan(plan, descriptor_set, blocks)

        self.static_constants_valid = False

        if self.submit_on_record:
            self.submit()       

    def submit(self, instance_count: int = None, stream_index: int = -2) -> None:
        """Submit the command list to the specified device with additional data to
        append to the front of the command list.
        
        Parameters:
        device_index (int): The device index to submit the command list to.\
                Default is 0.
        data (bytes): The additional data to append to the front of the command list.
        """

        if instance_count is None:
            instance_count = 1

        if not self.uniform_constants_valid:
            if len(self.pc_builder.element_map) > 0:
                self.pc_builder.prepare(instance_count)

                for key, value in self.pc_values.items():
                    self.pc_builder[key] = value
            
            if len(self.uniform_builder.element_map) > 0:
                self.uniform_builder.prepare(instance_count)

                for key, value in self.uniform_values.items():
                    self.uniform_builder[key] = value
                
                for descriptor_set, offset, size in self.uniform_descriptors:
                    descriptor_set.bind_buffer(self.uniform_constants_buffer, 0, offset, size, 1)

                self.uniform_constants_buffer.write(self.uniform_builder.tobytes())
            
            self.uniform_constants_valid = True

        my_data = None

        if len(self.pc_builder.element_map) > 0:
            my_data = self.pc_builder.tobytes()

        super().submit(data=my_data, stream_index=stream_index, instance_count=instance_count)

        if self._reset_on_submit:
            self.reset()
    
    def submit_any(self, data: bytes = None, instance_count: int = None) -> None:
        self.submit(data=data, stream_index=-1, instance_count=instance_count)
    
    def iter_batched_params(self, mapping_function, param_iter, batch_size: int = 10):
        data = b""
        bsize = 0

        for param in param_iter:
            mapping_function(param)

            for pc_buffer in self.pc_buffers:
                data += pc_buffer.get_bytes()

            bsize += 1

            if bsize == batch_size:
                yield data
                data = b""
                bsize = 0
            
        if bsize > 0:
            yield data

__default_cmd_stream = None
__custom_stream = None

def default_cmd_stream() -> CommandStream:
    global __default_cmd_stream

    if __default_cmd_stream is None:
        __default_cmd_stream = CommandStream(reset_on_submit=True, submit_on_record=True)

    return __default_cmd_stream

def global_cmd_stream() -> CommandStream:
    global __custom_stream

    if __custom_stream is not None:
        return __custom_stream

    return default_cmd_stream()

def set_global_cmd_stream(cmd_list: CommandStream = None) -> CommandStream:
    global __custom_stream
    old_value = __custom_stream
    __custom_stream = cmd_list 
    return old_value