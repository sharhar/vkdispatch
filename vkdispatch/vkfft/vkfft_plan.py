from vkdispatch.backends.backend_selection import native

import vkdispatch as vd

from typing import List
from typing import Tuple

from vkdispatch.base.errors import check_for_errors

from ..base.context import get_context, Context, Handle

class VkFFTPlan(Handle):
    context: Context

    def __init__(self, 
                 shape: Tuple[int, ...], 
                 do_r2c: bool = False, 
                 axes: List[int] = None, 
                 normalize: bool = False, 
                 padding: List[Tuple[int, int]] = None, 
                 pad_frequency_domain: bool = False,
                 kernel_count: int = 0,
                 input_shape: Tuple[int, ...] = None,
                 input_type: vd.dtype = None,
                 kernel_convolution: bool = False,
                 conjugate_convolution: bool = False,
                 convolution_features: int = 0,
                 num_batches: int = 1,
                 single_kernel_multiple_batches: bool = False,
                 keep_shader_code: bool = False):
        super().__init__()

        if len(shape) == 0 or len(shape) > 3:
            raise ValueError("Shape must be 1D, 2D, or 3D!")

        self.shape = shape
        self.do_r2c = do_r2c

        self.mem_size = vd.complex64.item_size
        for dim in shape:
            self.mem_size *= dim

        if axes is None:
            axes = [0, 1, 2]

        flipped_axes = [(len(self.shape) - 1)-a for a in axes]

        if padding is None:
            pad_left = [0, 0, 0]
            pad_right = [0, 0, 0]
        else:
            pad_left = [0, 0, 0]
            pad_right = [0, 0, 0]

            for i, padd in enumerate(padding):
                pad_left[(len(self.shape) - 1)-i] = padd[0]
                pad_right[(len(self.shape) - 1)-i] = padd[1]

        input_size = 0

        if input_shape is not None:
            input_buffer_type = vd.complex64 if input_type is None else input_type

            input_size = input_buffer_type.item_size
            for dim in input_shape:
                input_size *= dim

        handle = native.stage_fft_plan_create(
            self.context._handle, 
            list(reversed(self.shape)), 
            [axis for axis in flipped_axes if axis >= 0 and axis < 3], 
            self.mem_size, 
            do_r2c, 
            normalize,
            pad_left,
            pad_right,
            pad_frequency_domain,
            kernel_count,
            kernel_convolution,
            conjugate_convolution,
            convolution_features,
            input_size,
            num_batches,
            single_kernel_multiple_batches,
            keep_shader_code
        )
        check_for_errors()

        self.register_handle(handle)

    def _destroy(self):
        native.stage_fft_plan_destroy(self._handle)
        check_for_errors()

    def __del__(self):
        self.destroy()

    def record(self, graph: vd.CommandGraph, buffer: vd.Buffer, inverse: bool = False, kernel: vd.Buffer = None, input: vd.Buffer = None):
        native.stage_fft_record(
            graph._handle, 
            self._handle, 
            buffer._handle, 
            1 if inverse else -1, 
            kernel._handle if kernel is not None else 0,
            input._handle if input is not None else 0
        )
        check_for_errors()

    def record_forward(self, graph: vd.CommandGraph, buffer: vd.Buffer):
        self.record(graph, buffer, False)

    def record_inverse(self, graph: vd.CommandGraph, buffer: vd.Buffer):
        self.record(graph, buffer, True)
