import numpy as np

import vkdispatch_native

from .errors import check_for_errors

from .buffer import Buffer
from .context import get_context_handle
from .command_list import CommandList

from .dtype import dtype, complex64, to_numpy_dtype, from_numpy_dtype

from typing import List
from typing import Tuple

class FFTPlan:
    def __init__(self, 
                 shape: Tuple[int, ...], 
                 do_r2c: bool = False, 
                 axes: List[int] = None, 
                 normalize: bool = False, 
                 padding: List[Tuple[int, int]] = None, 
                 pad_frequency_domain: bool = False,
                 kernel_count: int = 0,
                 input_shape: Tuple[int, ...] = None,
                 input_type: dtype = None,
                 kernel_convolution: bool = False,
                 conjugate_convolution: bool = False,
                 convolution_features: int = 0):
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
                input_buffer_type = np.dtype(to_numpy_dtype(input_type))

            input_size = np.prod(input_shape) * input_buffer_type.itemsize

        self._handle = vkdispatch_native.stage_fft_plan_create(
            get_context_handle(), 
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
            input_size
        )
        check_for_errors()

    def record(self, command_list: CommandList, buffer: Buffer, inverse: bool = False, kernel: Buffer = None, input: Buffer = None):
        vkdispatch_native.stage_fft_record(
            command_list._handle, 
            self._handle, 
            buffer._handle, 
            1 if inverse else -1, 
            kernel._handle if kernel is not None else 0,
            input._handle if input is not None else 0
        )
        check_for_errors()

    def record_forward(self, command_list: CommandList, buffer: Buffer):
        self.record(command_list, buffer, False)

    def record_inverse(self, command_list: CommandList, buffer: Buffer):
        self.record(command_list, buffer, True)

