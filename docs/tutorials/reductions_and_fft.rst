Reductions and FFT Workflows
============================

This page covers common high-level numeric workflows in vkdispatch:

* reductions with ``vd.reduce``
* Fourier transforms with ``vd.fft``
* VkFFT-backed transforms with ``vd.vkfft``

FFT Subsystem Overview
----------------------

vkdispatch provides two FFT backends:

* ``vd.fft``: vkdispatch-generated shaders (runtime code generation).
* ``vd.vkfft``: VkFFT-backed plan execution.

Use ``vd.fft`` when you want shader-level customization and fusion through mapping
hooks (``input_map``, ``output_map``, ``kernel_map``). Use ``vd.vkfft`` when you want
the VkFFT path with plan caching and a similar high-level API.

Reduction Basics
----------------

Use ``@vd.reduce.reduce`` for pure binary reductions:

.. code-block:: python

   import numpy as np
   import vkdispatch as vd
   from vkdispatch.codegen.abreviations import *

   @vd.reduce.reduce(0)
   def sum_reduce(a: f32, b: f32) -> f32:
       return a + b

   arr = np.random.rand(4096).astype(np.float32)
   buf = vd.asbuffer(arr)
   out = sum_reduce(buf).read(0)

   print("GPU sum:", float(out[0]))
   print("CPU sum:", float(arr.sum(dtype=np.float32)))

Mapped Reductions
-----------------

Use ``@vd.reduce.map_reduce`` when you want a map stage before reduction:

.. code-block:: python

   import vkdispatch.codegen as vc

   @vd.reduce.map_reduce(vd.reduce.SubgroupAdd)
   def l2_energy_map(buffer: Buff[f32]) -> f32:
       idx = vd.reduce.mapped_io_index()
       v = buffer[idx]
       return v * v

   energy_buf = l2_energy_map(buf)
   energy = energy_buf.read(0)[0]

This pattern is useful for sums of transformed values (norms, weighted sums, etc.).

FFT with ``vd.fft``
-------------------

The ``vd.fft`` module dispatches vkdispatch-generated FFT shaders.

.. code-block:: python

   import numpy as np
   import vkdispatch as vd

   complex_signal = (
       np.random.rand(256) + 1j * np.random.rand(256)
   ).astype(np.complex64)

   fft_buf = vd.asbuffer(complex_signal)

   vd.fft.fft(fft_buf)
   freq = fft_buf.read(0)

   vd.fft.ifft(fft_buf)
   recovered = fft_buf.read(0)

   print(np.allclose(recovered, complex_signal, atol=1e-3))

By default, inverse transforms use normalization (``normalize=True`` in ``vd.fft.ifft``).
Set ``normalize=False`` when you need raw inverse scaling behavior.

To inspect generated FFT shaders, use:

.. code-block:: python

   vd.fft.fft(fft_buf, print_shader=True)

Axis and Dimensionality
-----------------------

FFT routines accept an ``axis`` argument for explicit axis control and provide ``fft2``
and ``fft3`` convenience functions.

.. code-block:: python

   # Strided FFT over the second axis of a 2D batch (from performance-test workflows).
   batch = (
       np.random.rand(8, 1024) + 1j * np.random.rand(8, 1024)
   ).astype(np.complex64)
   batch_buf = vd.asbuffer(batch)

   vd.fft.fft(batch_buf, axis=1)

   # 2D transform helper (last two axes).
   image = (
       np.random.rand(512, 512) + 1j * np.random.rand(512, 512)
   ).astype(np.complex64)
   image_buf = vd.asbuffer(image)
   vd.fft.fft2(image_buf)
   vd.fft.ifft2(image_buf)

Real FFT (RFFT) helpers:

.. code-block:: python

   real_signal = np.random.rand(512).astype(np.float32)
   rbuf = vd.asrfftbuffer(real_signal)

   vd.fft.rfft(rbuf)
   spectrum = rbuf.read_fourier(0)

   vd.fft.irfft(rbuf)
   restored = rbuf.read_real(0)

   print(np.allclose(restored, real_signal, atol=1e-3))

Fusion with ``kernel_map`` (Frequency-Domain In-Register Ops)
--------------------------------------------------------------

``vd.fft.convolve`` can inject custom frequency-domain logic via ``kernel_map``.
Inside a kernel map callback, ``vd.fft.read_op()`` exposes the current FFT register
being processed.

.. code-block:: python

   import vkdispatch.codegen as vc

   @vd.map
   def scale_spectrum(scale_factor: vc.Var[vc.f32]):
       op = vd.fft.read_op()
       op.register[:] = op.register * scale_factor

   # Fused forward FFT + frequency scaling + inverse FFT
   vd.fft.convolve(fft_buf, np.float32(0.5), kernel_map=scale_spectrum)

This pattern avoids a separate full-buffer dispatch for many pointwise spectral
operations.

Input/Output Mapping for Padded or Sparse Regions
-------------------------------------------------

For advanced workflows (for example padded 2D cross-correlation), use ``input_map`` and
``output_map`` to remap FFT I/O indices and ``input_signal_range`` to skip inactive
regions.

.. code-block:: python

   import vkdispatch.codegen as vc

   def padded_axis_fft(buffer: vd.Buffer, signal_cols: int):
       # Example expects buffer shape: (batch, rows, cols)
       trimmed_shape = (buffer.shape[0], signal_cols, buffer.shape[2])

       def remap(io_index: vc.ShaderVariable):
           return vc.unravel_index(
               vc.ravel_index(io_index, trimmed_shape).to_register(),
               buffer.shape
           )

       @vd.map
       def input_map(input_buffer: vc.Buffer[vc.c64]):
           op = vd.fft.read_op()
           op.read_from_buffer(input_buffer, io_index=remap(op.io_index))

       @vd.map
       def output_map(output_buffer: vc.Buffer[vc.c64]):
           op = vd.fft.write_op()
           op.write_to_buffer(output_buffer, io_index=remap(op.io_index))

       vd.fft.fft(
           buffer,
           buffer,
           buffer_shape=trimmed_shape,
           axis=1,
           input_map=input_map,
           output_map=output_map,
           input_signal_range=(0, signal_cols),
       )

Transposed Kernel Path for 2D Convolution
-----------------------------------------

When convolving along a strided axis, pre-transposing kernel layout can improve access
patterns. ``vd.fft`` provides helper APIs used by the benchmark suite:

.. code-block:: python

   # signal_buf and kernel_buf are complex buffers with compatible FFT shapes.
   transposed_size = vd.fft.get_transposed_size(signal_buf.shape, axis=1)
   kernel_t = vd.Buffer((transposed_size,), vd.complex64)

   vd.fft.transpose(kernel_buf, axis=1, out_buffer=kernel_t)

   vd.fft.fft(signal_buf)
   vd.fft.convolve(signal_buf, kernel_t, axis=1, transposed_kernel=True)
   vd.fft.ifft(signal_buf)

Low-Level Procedural FFT Generation with ``fft_context``
--------------------------------------------------------

For full control over read/compute/write staging, build FFT shaders procedurally using
``vd.fft.fft_context`` and iterators from ``vd.fft``:

.. code-block:: python

   import vkdispatch.codegen as vc

   with vd.fft.fft_context(buffer_shape=(1024,), axis=0) as ctx:
       args = ctx.declare_shader_args([vc.Buffer[vc.c64]])

       for read_op in vd.fft.global_reads_iterator(ctx.registers):
           read_op.read_from_buffer(args[0])

       ctx.execute(inverse=False)

       for write_op in vd.fft.global_writes_iterator(ctx.registers):
           write_op.write_to_buffer(args[0])

   fft_kernel = ctx.get_callable()
   fft_kernel(fft_buf)

FFT with ``vd.vkfft``
---------------------

``vd.vkfft`` exposes a similar API but routes operations through VkFFT plan objects
with internal plan caching.

.. code-block:: python

   vkfft_buf = vd.asbuffer(complex_signal.copy())
   vd.vkfft.fft(vkfft_buf)
   vd.vkfft.ifft(vkfft_buf)
   print(np.allclose(vkfft_buf.read(0), complex_signal, atol=1e-3))

After large parameter sweeps, clearing cached plans can be helpful:

.. code-block:: python

   vd.vkfft.clear_plan_cache()
   vd.fft.cache_clear()

Convolution Helpers
-------------------

vkdispatch also includes FFT-based convolution helpers:

* ``vd.fft.convolve`` / ``vd.fft.convolve2D`` / ``vd.fft.convolve2DR``
* ``vd.vkfft.convolve2D`` and ``vd.vkfft.transpose_kernel2D``

These APIs are most useful when you repeatedly convolve signals/images with known
kernel layouts.

Reduction and FFT API Reference
-------------------------------

See the :doc:`Full Python API Reference <../python_api>` for complete API details on:

* ``vkdispatch.reduce``
* ``vkdispatch.fft``
* ``vkdispatch.vkfft``
