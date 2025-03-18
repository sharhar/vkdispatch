import numpy as np

import vkdispatch_native

import vkdispatch as vd

from typing import List
from typing import Tuple

class VkFFTPlan:
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
                 single_kernel_multiple_batches: bool = False):
        assert len(shape) > 0 and len(shape) < 4, "shape must be 1D, 2D, or 3D"

        self.shape = shape
        self.do_r2c = do_r2c

        self.mem_size = (
            np.prod(shape) * np.dtype(np.complex64).itemsize
        )  # currently only support complex64

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
            input_buffer_type = np.dtype(np.complex64)

            if input_type is not None:
                input_buffer_type = np.dtype(vd.to_numpy_dtype(input_type))

            input_size = np.prod(input_shape) * input_buffer_type.itemsize

        self._handle = vkdispatch_native.stage_fft_plan_create(
            vd.get_context_handle(), 
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
            single_kernel_multiple_batches
        )
        vd.check_for_errors()

    def record(self, command_list: vd.CommandList, buffer: vd.Buffer, inverse: bool = False, kernel: vd.Buffer = None, input: vd.Buffer = None):
        vkdispatch_native.stage_fft_record(
            command_list._handle, 
            self._handle, 
            buffer._handle, 
            1 if inverse else -1, 
            kernel._handle if kernel is not None else 0,
            input._handle if input is not None else 0
        )
        vd.check_for_errors()

    def record_forward(self, command_list: vd.CommandList, buffer: vd.Buffer):
        self.record(command_list, buffer, False)

    def record_inverse(self, command_list: vd.CommandList, buffer: vd.Buffer):
        self.record(command_list, buffer, True)

