from typing import Any
from typing import List
from typing import Dict
from typing import Tuple, Optional

import uuid
import threading

import vkdispatch as vd
import vkdispatch.codegen as vc

from vkdispatch.base.command_list import CommandList
from vkdispatch.base.compute_plan import ComputePlan
from vkdispatch.base.descriptor_set import DescriptorSet

from .buffer_builder import BufferUsage
from .buffer_builder import BufferBuilder

import dataclasses

def _runtime_supports_push_constants() -> bool:
    return not (vd.is_cuda() or vd.is_opencl())

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

class CommandGraph(CommandList):
    """
    A high-level abstraction over ``CommandList`` that manages resource binding and push constants automatically.

    Unlike a raw ``CommandList``, a ``CommandGraph`` tracks variable state and handles the 
    complexities of ``BufferBuilder`` for push constants and uniform buffers. It serves 
    as the default recording target for shader functions.

    :param reset_on_submit: If True, the graph clears its recorded commands immediately after submission.
    :type reset_on_submit: bool
    :param submit_on_record: If True, commands are submitted to the GPU immediately upon recording 
                             (simulating immediate mode execution).
    :type submit_on_record: bool
    """

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

    uniform_descriptors: List[Tuple[DescriptorSet, int, int]]
    recorded_descriptor_sets: List[DescriptorSet]

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
        self.recorded_descriptor_sets = []

        self._reset_on_submit = reset_on_submit
        self.submit_on_record = submit_on_record

        self.uniform_constants_size = 4096
        self.uniform_constants_buffer = vd.Buffer(shape=(4096,), var_type=vd.uint32) # Create a base static constants buffer at size 4k bytes

    def _ensure_uniform_constants_capacity(self, uniform_word_size: int) -> None:
        if uniform_word_size <= self.uniform_constants_size:
            return

        # Grow exponentially to reduce reallocation churn for larger UBO layouts.
        self.uniform_constants_size = max(uniform_word_size, self.uniform_constants_size * 2)
        self.uniform_constants_buffer = vd.Buffer(shape=(self.uniform_constants_size,), var_type=vd.uint32)

    def _prepare_submission_state(self, instance_count: int) -> None:
        if len(self.pc_builder.element_map) > 0 and (
                self.pc_builder.instance_count != instance_count or not self.buffers_valid
            ):

            assert _runtime_supports_push_constants(), (
                "Push constants not supported for backends without push-constant support "
                "(CUDA/OpenCL). Use UBO-backed variables instead."
            )

            self.pc_builder.prepare(instance_count)

            for key, value in self.pc_values.items():
                self.pc_builder[key] = value

        if len(self.uniform_builder.element_map) > 0 and not self.buffers_valid:
            self.uniform_builder.prepare(1)

            for key, value in self.uniform_values.items():
                self.uniform_builder[key] = value

            uniform_word_size = (self.uniform_builder.instance_bytes + 3) // 4
            
            uniform_buffer = None

            if vd.get_cuda_capture() is not None:
                uniform_buffer = vd.Buffer(shape=(uniform_word_size,), var_type=vd.uint32)
            else:
                self._ensure_uniform_constants_capacity(uniform_word_size)
                uniform_buffer = self.uniform_constants_buffer

            for descriptor_set, offset, size in self.uniform_descriptors:
                descriptor_set.bind_buffer(uniform_buffer, 0, offset, size, True, write_access=False)

            uniform_buffer.write(self.uniform_builder.tobytes())

            if vd.get_cuda_capture() is not None:
                vd.get_cuda_capture().add_uniform_buffer(uniform_buffer)

        if not self.buffers_valid:
            self.buffers_valid = True

    def prepare_for_cuda_graph_capture(self, instance_count: int = None) -> None:
        """Initialize internal data uploads before torch CUDA graph capture.

        This method performs one-time uniform/push-constant staging without submitting
        the command list, so only kernel launches are captured by ``torch.cuda.graph``.
        """
        if instance_count is None:
            instance_count = 1

        self._prepare_submission_state(instance_count)

    def reset(self) -> None:
        """Reset the command graph by clearing the push constant buffer and descriptor
        set lists.
        """
        super().reset()

        self.pc_builder.reset()
        self.uniform_builder.reset()

        for descriptor_set in self.recorded_descriptor_sets:
            descriptor_set.destroy()
        
        self.pc_values.clear()
        self.uniform_values.clear()
        self.name_to_pc_key_dict.clear()
        self.queued_pc_values.clear()
        self.uniform_descriptors.clear()
        self.recorded_descriptor_sets.clear()

        self.buffers_valid = False

    def _destroy(self) -> None:
        self.reset()
        super()._destroy()
    
    def bind_var(self, name: str):
        if not _runtime_supports_push_constants():
            raise RuntimeError(
                "CommandGraph.bind_var() is disabled for backends without push-constant "
                "support (CUDA/OpenCL). Pass Variable values directly at shader invocation."
            )

        def register_var(key: Tuple[str, str]):
            if not name in self.name_to_pc_key_dict.keys():
                self.name_to_pc_key_dict[name] = []

            self.name_to_pc_key_dict[name].append(key)

        return register_var
    
    def set_var(self, name: str, value: Any):
        if not _runtime_supports_push_constants():
            raise RuntimeError(
                "CommandGraph.set_var() is disabled for backends without push-constant "
                "support (CUDA/OpenCL). Pass Variable values directly at shader invocation."
            )

        if name not in self.name_to_pc_key_dict.keys():
            raise ValueError("Variable not bound!")
        
        for key in self.name_to_pc_key_dict[name]:
            self.queued_pc_values[key] = value
    
    def record_shader(self, 
                      plan: ComputePlan,
                      shader_description: vc.ShaderDescription, 
                      exec_limits: Tuple[int, int, int], 
                      blocks: Tuple[int, int, int],
                      bound_buffers: List[BufferBindInfo],
                      bound_samplers: List[ImageBindInfo],
                      uniform_values: Dict[str, Any] = {},
                      pc_values: Dict[str, Any] = {},
                      shader_uuid: str = None
                    ) -> None:
        """
        Internal method to record a high-level shader execution.

        This method handles the creation of ``DescriptorSet`` objects, binding of buffers 
        and images, and populating push constant/uniform data before calling the base 
        ``record_compute_plan``.

        :param plan: The compute plan to execute.
        :param shader_description: Metadata about the shader source and layout.
        :param exec_limits: The execution limits (grid size) in x, y, z.
        :param blocks: The number of workgroups to dispatch.
        :param bound_buffers: List of buffers to bind.
        :param bound_samplers: List of images/samplers to bind.
        :param uniform_values: Dictionary of values for uniform buffer objects.
        :param pc_values: Dictionary of values for push constants.
        :param shader_uuid: Unique identifier for this shader instance (for caching).
        """

        descriptor_set = DescriptorSet(plan)
        self.recorded_descriptor_sets.append(descriptor_set)

        if shader_uuid is None:
            shader_uuid = shader_description.name + "_" + str(uuid.uuid4())

        if (not _runtime_supports_push_constants()) and len(pc_values) > 0:
            raise RuntimeError(
                "Push-constant Variable payloads are disabled for backends without "
                "push-constant support (CUDA/OpenCL). "
                "Variable values must be UBO-backed and provided at shader invocation."
            )

        if len(shader_description.pc_structure) != 0:
            if not _runtime_supports_push_constants():
                raise RuntimeError(
                    "Kernels should not emit push-constant layouts for backends without "
                    "push-constant support (CUDA/OpenCL). Use UBO-backed variables."
                )
            self.pc_builder.register_struct(shader_uuid, shader_description.pc_structure)

        uniform_field_names = {elem.name for elem in shader_description.uniform_structure}
        resolved_uniform_values: Dict[Tuple[str, str], Any] = {}

        if shader_description.exec_count_name is not None:
            resolved_uniform_values[(shader_uuid, shader_description.exec_count_name)] = [
                exec_limits[0],
                exec_limits[1],
                exec_limits[2],
                0,
            ]

        for buffer_bind_info in bound_buffers:
            descriptor_set.bind_buffer(
                buffer=buffer_bind_info.buffer,
                binding=buffer_bind_info.binding,
                read_access=buffer_bind_info.read_access,
                write_access=buffer_bind_info.write_access,
            )
            
            if buffer_bind_info.shape_name in uniform_field_names:
                resolved_uniform_values[(shader_uuid, buffer_bind_info.shape_name)] = buffer_bind_info.buffer.shader_shape
        
        for sampler_bind_info in bound_samplers:
            descriptor_set.bind_sampler(
                sampler_bind_info.sampler,
                sampler_bind_info.binding,
                read_access=sampler_bind_info.read_access,
                write_access=sampler_bind_info.write_access
            )

        for key, value in uniform_values.items():
            resolved_uniform_values[(shader_uuid, key)] = value

        if len(shader_description.uniform_structure) > 0:
            uniform_offset, uniform_range = self.uniform_builder.register_struct(shader_uuid, shader_description.uniform_structure)
            self.uniform_descriptors.append((descriptor_set, uniform_offset, uniform_range))

        for key, value in resolved_uniform_values.items():
            self.uniform_values[key] = value
        
        for key, value in pc_values.items():
            self.pc_values[(shader_uuid, key)] = value

        super().record_compute_plan(plan, descriptor_set, blocks)

        self.buffers_valid = False
        
        if self.submit_on_record:
            self.submit()
    
    def submit(
        self,
        instance_count: int = None,
        queue_index: int = -2
    ) -> None:
        """Submit the command list to the specified device with additional data to
        append to the front of the command list.
        
        Parameters:
        device_index (int): The device index to submit the command list to.
                Default is 0.
        data (bytes): The additional data to append to the front of the command list.
        """

        if instance_count is None:
            instance_count = 1

        self._prepare_submission_state(instance_count)

        for key, val in self.queued_pc_values.items():
            self.pc_builder[key] = val
        
        my_data = None

        if len(self.pc_builder.element_map) > 0:
            my_data = self.pc_builder.tobytes()

        super().submit(
            data=my_data,
            queue_index=queue_index,
            instance_count=instance_count,
            cuda_stream=None,
        )

        if self._reset_on_submit:
            self.reset()
    
    def submit_any(self, instance_count: int = None) -> None:
        self.submit(instance_count=instance_count, queue_index=-1)

_global_graph = threading.local()

def _get_global_graph() -> Optional[CommandGraph]:
    return getattr(_global_graph, 'custom_graph', None)

def default_graph() -> CommandGraph:
    if not hasattr(_global_graph, 'default_graph'):
        _global_graph.default_graph = CommandGraph(reset_on_submit=True, submit_on_record=True)

    return _global_graph.default_graph

def global_graph() -> CommandGraph:
    custom_graph = _get_global_graph()

    if custom_graph is not None:
        return custom_graph

    return default_graph()

def set_global_graph(graph: CommandGraph = None) -> CommandGraph:
    if graph is None:
        _global_graph.custom_graph = None
        return

    assert _get_global_graph() is None, "A global CommandGraph is already set for the current thread!"
    _global_graph.custom_graph = graph
